//
//  SpliceKitTimelinePlayheadOverlay.m
//  120Hz cosmetic playhead overlay for timeline playback.
//
//  Problem
//  -------
//  FCP's playhead updates are driven by FFContext.time changes
//  (timeRateChangedForContext:), not by CVDisplayLink. On a 24p or 30p
//  project the playhead steps at the source frame rate regardless of the
//  display's refresh rate. On a 120Hz ProMotion panel that reads as stutter.
//
//  Approach
//  --------
//  Don't force TimelineKit to redraw at 120Hz — filmstrips and waveforms
//  re-layout on every step, which is expensive. Instead, draw a cosmetic
//  vertical line as an overlay layer on top of the timeline's content layer
//  and move it at 120Hz by extrapolating the playhead forward from the last
//  observed (time, wallClock, rate) triple.
//
//  Extrapolation
//    t_now = t_last + (CACurrentMediaTime() - wallClock_last) * rate
//    x_now = [timelineView locationRangeForTime: t_now].location
//
//  Observation
//    Swizzle -[TLKTimelineView _setPlayheadTime_NoKVO:animate:] to capture
//    every real playhead update. Read rate from the FFContext via
//    [timelineModule context].rate.
//
//  We hide Apple's real playhead layer while our overlay is active so the
//  user sees one smooth line, not one smooth + one stuttery. On pause we
//  restore the real layer and hide ours.
//

#import "SpliceKit.h"
#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>
#import <CoreMedia/CoreMedia.h>
#import <objc/runtime.h>
#import <objc/message.h>

// objc_msgSend_stret is x86-only. On arm64, struct returns use plain msgSend.
#if defined(__x86_64__)
#define PO_STRET objc_msgSend_stret
#else
#define PO_STRET objc_msgSend
#endif

// Opaque CMTime mirror — matches sizeof(CMTime) = 24 with 8-byte alignment.
typedef struct __attribute__((aligned(8))) {
    int64_t value;
    int32_t timescale;
    uint32_t flags;
    int64_t epoch;
} PO_CMTime;

// TLKTimelineView.locationRangeForTime: returns _TLKRange { double location; double length; }
typedef struct {
    double location;
    double length;
} PO_TLKRange;

static NSString * const kDefTimelinePlayheadOverlay = @"SpliceKitTimelinePlayheadOverlay";

static BOOL        sOverlayInstalled = NO;
static BOOL        sIsPlaying = NO;
static CAShapeLayer *sOverlayLayer = nil;
static __weak NSView *sTimelineViewWeak = nil;     // FFProTimelineView / TLKTimelineView
static __weak CALayer *sRealPlayheadLayerWeak = nil;
static BOOL        sRealPlayheadWasHidden = NO;
static CADisplayLink *sDisplayLink = nil;

// When we pause Apple's TLKScrollingTimeline during playback to stop its
// 30Hz step-based auto-scroll from fighting our 120Hz smooth scroll, we
// need to remember which scrollingTimeline we paused and its prior state
// so we can restore on playback end.
static __weak id sPausedScrollingTimelineWeak = nil;
static __weak NSView *sPausedTimelineViewWeak = nil;
static BOOL      sScrollingTimelineWasPaused = NO;

// Last observed playhead state from the swizzle.
// Guarded by sObservedLock.
static PO_CMTime  sObservedTime = {0};
static CFTimeInterval sObservedWall = 0;
static double     sObservedRate = 0.0;
static os_unfair_lock sObservedLock = OS_UNFAIR_LOCK_INIT;

// The exact TLKTimelineView that received the most recent
// _setPlayheadTime_NoKVO:animate: swizzled call. Preferred over window-walking
// in the display-link tick because it's the object actually receiving
// playhead updates — no risk of ending up on a wrong timeline view when the
// dual-timeline panel is open. Set under sObservedLock.
static __weak NSView *sSwizzleCapturedViewWeak = nil;

static IMP sOrigSetPlayheadTimeNoKVO = NULL;

// ---- Helpers ----

static NSView *PO_findFFProTimelineView(NSView *root, NSInteger depth) {
    if (!root || depth > 20) return nil;
    if ([NSStringFromClass([root class]) isEqualToString:@"FFProTimelineView"]) return root;
    for (NSView *sub in root.subviews) {
        NSView *hit = PO_findFFProTimelineView(sub, depth + 1);
        if (hit) return hit;
    }
    return nil;
}

static NSView *PO_currentTimelineView(void) {
    // Prefer the view captured in the swizzle — it's the one actually
    // receiving playhead updates right now, even across dual-timeline /
    // focus changes.
    NSView *fromSwizzle = sSwizzleCapturedViewWeak;
    if (fromSwizzle && fromSwizzle.window) {
        sTimelineViewWeak = fromSwizzle;
        return fromSwizzle;
    }
    NSView *existing = sTimelineViewWeak;
    if (existing && existing.window) return existing;
    for (NSWindow *w in [NSApp windows]) {
        NSString *cn = NSStringFromClass([w class]);
        if (![cn containsString:@"PEWindow"]) continue;
        NSView *hit = PO_findFFProTimelineView(w.contentView, 0);
        if (hit) {
            sTimelineViewWeak = hit;
            return hit;
        }
    }
    return nil;
}

// Try a few ivar names Apple has used for the playhead layer on TLKTimelineView.
static CALayer *PO_findRealPlayheadLayer(NSView *timelineView) {
    if (!timelineView) return nil;
    NSArray *keys = @[@"playheadMarker", @"_playheadMarker", @"playhead",
                      @"_playhead", @"playheadLayer", @"_playheadLayer"];
    for (NSString *key in keys) {
        @try {
            id v = [timelineView valueForKey:key];
            if ([v isKindOfClass:[CALayer class]]) {
                return v;
            }
        } @catch (NSException *e) {
            // valueForKey: is strict — swallow and continue
        }
    }
    return nil;
}

// Query the active FFContext's current rate. During playback this is typically
// 1.0 (or the L/J speed); while paused it's 0.0.
static double PO_currentRate(void) {
    id tm = SpliceKit_getActiveTimelineModule();
    if (!tm) return 0.0;
    SEL ctxSel = NSSelectorFromString(@"context");
    if (![tm respondsToSelector:ctxSel]) return 0.0;
    id ctx = ((id (*)(id, SEL))objc_msgSend)(tm, ctxSel);
    if (!ctx) return 0.0;
    SEL rateSel = NSSelectorFromString(@"rate");
    if (![ctx respondsToSelector:rateSel]) return 0.0;
    return ((double (*)(id, SEL))objc_msgSend)(ctx, rateSel);
}

// Convert an extrapolated CMTime to an x coordinate in the timeline view's
// own coordinate space, via -[TLKTimelineView locationRangeForTime:].
static BOOL PO_xForTime(NSView *timelineView, PO_CMTime t, double *outX) {
    if (!timelineView || !outX) return NO;
    SEL sel = @selector(locationRangeForTime:);
    if (![timelineView respondsToSelector:sel]) return NO;
    PO_TLKRange range = {0};
    @try {
        range = ((PO_TLKRange (*)(id, SEL, PO_CMTime))PO_STRET)(timelineView, sel, t);
    } @catch (NSException *e) {
        return NO;
    }
    *outX = range.location;
    return YES;
}

// ---- Swizzled -[TLKTimelineView _setPlayheadTime_NoKVO:animate:] ----

static void SpliceKit_swizzled_setPlayheadTimeNoKVO(id self_, SEL _cmd,
                                                     PO_CMTime time, BOOL animate) {
    // Call original FIRST so the real UI updates as normal. Then capture.
    ((void (*)(id, SEL, PO_CMTime, BOOL))sOrigSetPlayheadTimeNoKVO)(self_, _cmd, time, animate);

    double rate = PO_currentRate();
    CFTimeInterval now = CACurrentMediaTime();

    os_unfair_lock_lock(&sObservedLock);
    sObservedTime = time;
    sObservedWall = now;
    sObservedRate = rate;
    // Capture the actual TLKTimelineView receiving the update — the tick
    // should prefer this over window-walking so dual-timeline / focus
    // changes don't land us on a stale view.
    if ([self_ isKindOfClass:[NSView class]]) {
        sSwizzleCapturedViewWeak = (NSView *)self_;
    }
    os_unfair_lock_unlock(&sObservedLock);
}

// ---- Overlay layer management ----

static void PO_attachOverlayLayerIfNeeded(NSView *timelineView) {
    if (!timelineView) return;
    [timelineView setWantsLayer:YES];
    CALayer *host = timelineView.layer;
    if (!host) return;

    if (!sOverlayLayer) {
        sOverlayLayer = [CAShapeLayer layer];
        sOverlayLayer.name = @"SpliceKitPlayheadOverlay";
        sOverlayLayer.anchorPoint = CGPointMake(0.5, 0.0);
        sOverlayLayer.zPosition = 999.0; // above almost everything
        sOverlayLayer.strokeColor = [NSColor colorWithCalibratedRed:1.0 green:0.85 blue:0.1 alpha:0.95].CGColor;
        sOverlayLayer.fillColor   = nil;
        sOverlayLayer.lineWidth   = 1.5;
        sOverlayLayer.hidden      = YES;
        sOverlayLayer.actions = @{
            @"position": [NSNull null],
            @"bounds":   [NSNull null],
            @"path":     [NSNull null],
            @"hidden":   [NSNull null],
        };
    }

    if (sOverlayLayer.superlayer != host) {
        [sOverlayLayer removeFromSuperlayer];
        [host addSublayer:sOverlayLayer];
    }
}

static void PO_updateOverlayPath(NSView *timelineView) {
    if (!sOverlayLayer || !timelineView) return;
    CGFloat h = timelineView.bounds.size.height;
    CGMutablePathRef p = CGPathCreateMutable();
    CGPathMoveToPoint(p, NULL, 0.0, 0.0);
    CGPathAddLineToPoint(p, NULL, 0.0, h);
    sOverlayLayer.path = p;
    CGPathRelease(p);
    sOverlayLayer.bounds = CGRectMake(-1.0, 0.0, 2.0, h);
}

// Restore the native playhead that the overlay currently owns.  Keeping this
// in one helper is important because FCP can replace TLKTimelineView while
// playback is still running (project switches, workspace changes, dual
// timeline focus changes).  In that case the old native layer must not remain
// hidden after our cosmetic layer moves to the new view.
static void PO_restoreRealPlayhead(void) {
    CALayer *real = sRealPlayheadLayerWeak;
    if (real) real.hidden = sRealPlayheadWasHidden;
    sRealPlayheadLayerWeak = nil;
    sRealPlayheadWasHidden = NO;
}

// Stop presenting the cosmetic playhead and immediately fall back to FCP's
// native one.  Every failure path in the display-link tick uses this so a
// missing/stale observation can never leave both playheads invisible.
static void PO_deactivateOverlayVisual(void) {
    if (sOverlayLayer) sOverlayLayer.hidden = YES;
    PO_restoreRealPlayhead();
}

static void PO_restorePausedScrollingTimeline(void) {
    id paused = sPausedScrollingTimelineWeak;
    if (paused) {
        @try {
            SEL pausedSet = NSSelectorFromString(@"setPaused:");
            if ([paused respondsToSelector:pausedSet]) {
                ((void (*)(id, SEL, BOOL))objc_msgSend)(paused, pausedSet,
                                                        sScrollingTimelineWasPaused);
            }
        } @catch (...) {}
    }
    sPausedScrollingTimelineWeak = nil;
    sPausedTimelineViewWeak = nil;
    sScrollingTimelineWasPaused = NO;
}

// Transfer visual ownership to timelineView only after the overlay has a
// valid position.  This also reattaches the layer when FCP swaps timeline
// views mid-playback, which was the main cause of the yellow line vanishing.
static BOOL PO_activateOverlayVisual(NSView *timelineView) {
    if (!timelineView) return NO;
    if (sPausedScrollingTimelineWeak && timelineView != sPausedTimelineViewWeak) {
        // The paused scroller belongs to the timeline that just went away.
        // Restore it; the replacement view keeps its native scroll behavior.
        PO_restorePausedScrollingTimeline();
    }
    PO_attachOverlayLayerIfNeeded(timelineView);
    if (!sOverlayLayer || sOverlayLayer.superlayer != timelineView.layer) {
        PO_deactivateOverlayVisual();
        return NO;
    }

    CALayer *real = PO_findRealPlayheadLayer(timelineView);
    if (real != sRealPlayheadLayerWeak) {
        PO_restoreRealPlayhead();
        sRealPlayheadLayerWeak = real;
        if (real) {
            sRealPlayheadWasHidden = real.hidden;
            real.hidden = YES;
        }
    } else if (real) {
        // TimelineKit can unhide its layer during an internal reload.
        real.hidden = YES;
    }

    sOverlayLayer.hidden = NO;
    return YES;
}

// ---- Display link tick ----

@interface SpliceKitPlayheadOverlayTarget : NSObject
@end

// Reads the user's "Continuous Scrolling" preference — the toggle that
// controls whether the timeline scrolls to keep the playhead centered
// during playback, or whether the playhead slides off to the right.
//
// This lives on TLKTimelineView as -scrollDuringPlayback (backed by the
// FFScrollDuringPlaybackKey NSUserDefaults key, wired from
// -[FFAnchoredTimelineModule updateTimelineScrollDuringPlaybackToMatchUserDefaults]).
//
// IMPORTANT: we deliberately do NOT use `keepsPlayheadCenteredDuringPlayback`
// here. That one looks like the right property by name, but it's a computed
// value driven by playback rate — it's NO at normal rate=1.0 and only
// flips YES during fast-forward / rewind. Using it as a gate made Smooth
// Scroll silently skip the 120Hz centered path for every normal playback.
static BOOL PO_scrollDuringPlayback(id timelineView) {
    if (!timelineView) return NO;
    @try {
        SEL sel = NSSelectorFromString(@"scrollDuringPlayback");
        if (![timelineView respondsToSelector:sel]) return NO;
        return ((BOOL (*)(id, SEL))objc_msgSend)(timelineView, sel);
    } @catch (...) {}
    return NO;
}

@implementation SpliceKitPlayheadOverlayTarget
- (void)tick:(CADisplayLink *)link {
    NSView *view = PO_currentTimelineView();
    if (!view || !sOverlayLayer) {
        PO_deactivateOverlayVisual();
        return;
    }

    os_unfair_lock_lock(&sObservedLock);
    PO_CMTime base = sObservedTime;
    CFTimeInterval baseWall = sObservedWall;
    double rate = sObservedRate;
    os_unfair_lock_unlock(&sObservedLock);

    // If nothing has been observed yet, we have no reference — skip this tick.
    if (base.timescale <= 0 || baseWall <= 0.0) {
        PO_deactivateOverlayVisual();
        return;
    }

    // Skip when paused: tick only needs to extrapolate during active playback,
    // and if we've gone idle the cached view may be mid-teardown. APPLE-MACOS-P
    // shows EXC_BAD_ACCESS deep inside locationRangeForTime: when this fires
    // against a view whose internals were freed.
    if (rate == 0.0) {
        PO_deactivateOverlayVisual();
        return;
    }

    // Stale observation: if we haven't seen a setPlayheadTime in a while, the
    // cached view may have been replaced (sequence change, dual-timeline
    // toggle). Bail out and wait for a fresh observation rather than calling
    // into a possibly-dead view.
    CFTimeInterval elapsed = CACurrentMediaTime() - baseWall;
    if (elapsed > 5.0) {
        PO_deactivateOverlayVisual();
        return;
    }

    // Extrapolate forward: t_now = base + (now - baseWall) * rate
    double extraSecs = elapsed * rate;
    PO_CMTime extrapolated = base;
    int64_t addValue = (int64_t)llround(extraSecs * (double)base.timescale);
    extrapolated.value += addValue;

    double x = 0.0;
    if (!PO_xForTime(view, extrapolated, &x) || !isfinite(x)) {
        PO_deactivateOverlayVisual();
        return;
    }

    // Set the new position before hiding the native playhead.  If FCP has
    // replaced the timeline view, activation reparents our layer atomically
    // from the user's perspective instead of leaving a blank interval.
    sOverlayLayer.position = CGPointMake(x, 0.0);
    if (!PO_activateOverlayVisual(view)) return;

    CGFloat h = view.bounds.size.height;
    if (sOverlayLayer.bounds.size.height != h) {
        PO_updateOverlayPath(view);
    }

    // ── Smooth centered-scroll path ──────────────────────────────────────
    // During Perf Mode playback (when the safety gate accepted), Apple's
    // TLKScrollingTimeline is paused and our tick is authoritative. Drive
    // the scroll directly on the clip view — that's what Apple's own
    // scrollPoint: ultimately does, minus any guards in
    // -[TLKTimelineView scrollTimelineToPoint:] that can short-circuit
    // when the tlkViewFlags re-entrancy bit happens to be set.
    NSRect vrect = NSZeroRect;
    @try {
        vrect = [view visibleRect];
    } @catch (...) {}

    BOOL drivingScroll = (sPausedScrollingTimelineWeak != nil &&
                          view == sPausedTimelineViewWeak);
    CGFloat overlayX = x;  // content-space x for the overlay line

    if (drivingScroll && vrect.size.width > 0.0) {
        CGFloat halfWidth = vrect.size.width * 0.5;
        CGFloat targetOriginX = x - halfWidth;

        // Clamp to content bounds so we never ask for a negative or
        // beyond-end origin (scrollPoint quietly does this too, but doing
        // it up front also gives us an accurate comparison to decide
        // whether we need to scroll at all).
        CGFloat contentWidth = view.bounds.size.width;
        CGFloat maxOriginX = MAX(0.0, contentWidth - vrect.size.width);
        if (targetOriginX < 0.0) targetOriginX = 0.0;
        if (targetOriginX > maxOriginX) targetOriginX = maxOriginX;

        if (fabs(vrect.origin.x - targetOriginX) > 0.25) {
            NSView *docSuper = view.superview;
            NSClipView *clipView = [docSuper isKindOfClass:[NSClipView class]]
                ? (NSClipView *)docSuper : nil;
            NSPoint newOrigin = NSMakePoint(targetOriginX, vrect.origin.y);

            [CATransaction begin];
            [CATransaction setDisableActions:YES];
            if (clipView) {
                // Low-level path: move the clip view's bounds origin and
                // reflect to the scroll view. Bypasses Apple's
                // scrollTimelineToPoint: reentrancy guard.
                [clipView setBoundsOrigin:newOrigin];
                [clipView.enclosingScrollView reflectScrolledClipView:clipView];
                // Update vrect so the center-pin math below uses the new origin.
                vrect.origin.x = targetOriginX;
            } else {
                // Fallback: standard scrollPoint on the timeline view.
                [view scrollPoint:newOrigin];
                vrect.origin.x = targetOriginX;
            }
            [CATransaction commit];

            // Diagnostic: log the first few ticks so we can verify the
            // scroll is actually moving. Throttle so we don't flood the log.
            static int sLogged = 0;
            if (sLogged < 5) {
                SpliceKit_log(@"[PlayheadOverlay] tick x=%.1f vrect.origin.x=%.1f → %.1f "
                              @"clipView=%@ contentW=%.0f viewportW=%.0f",
                              x, vrect.origin.x, targetOriginX,
                              clipView ? @"yes" : @"no",
                              contentWidth, vrect.size.width);
                sLogged++;
            }
        }

        // Leave overlayX at content-space x (the real playhead position).
        // When we successfully centered, x == vrect.origin.x + halfWidth
        // anyway — so the overlay naturally appears at screen center. When
        // the viewport was clamped (playhead near content start/end and we
        // couldn't scroll further), x stays at the actual playhead position
        // instead of lying about being in the middle.
    }

    sOverlayLayer.position = CGPointMake(overlayX, 0.0);
}
@end

static SpliceKitPlayheadOverlayTarget *sDisplayLinkTarget = nil;

static void PO_startDisplayLink(NSView *timelineView) {
    if (sDisplayLink) return;
    if (!sDisplayLinkTarget) sDisplayLinkTarget = [[SpliceKitPlayheadOverlayTarget alloc] init];

    CADisplayLink *link = nil;
    if ([timelineView.window respondsToSelector:@selector(displayLinkWithTarget:selector:)]) {
        link = [timelineView.window displayLinkWithTarget:sDisplayLinkTarget
                                                 selector:@selector(tick:)];
    }
    if (!link) {
        link = [NSScreen.mainScreen displayLinkWithTarget:sDisplayLinkTarget
                                                 selector:@selector(tick:)];
    }
    if (!link) return;
    [link addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSRunLoopCommonModes];
    sDisplayLink = link;
}

static void PO_stopDisplayLink(void) {
    [sDisplayLink invalidate];
    sDisplayLink = nil;
}

// ---- Play/pause transitions ----

static void PO_onPlaybackBegan(void) {
    sIsPlaying = YES;
    // A missed end notification from a previous playback must not carry
    // hidden-layer ownership into the new session.
    PO_deactivateOverlayVisual();
    PO_restorePausedScrollingTimeline();
    NSView *view = PO_currentTimelineView();
    if (!view) return;
    PO_attachOverlayLayerIfNeeded(view);
    PO_updateOverlayPath(view);

    // Always re-seed observation on playback begin so extrapolation starts
    // from the current-known state, not stale data from a previous session.
    SEL phSel = @selector(playheadTime);
    if ([view respondsToSelector:phSel]) {
        PO_CMTime t = ((PO_CMTime (*)(id, SEL))PO_STRET)(view, phSel);
        if (t.timescale > 0) {
            os_unfair_lock_lock(&sObservedLock);
            sObservedTime = t;
            sObservedWall = CACurrentMediaTime();
            sObservedRate = PO_currentRate();
            os_unfair_lock_unlock(&sObservedLock);
        }
    }

    // Respect the user's "Continuous Scrolling" preference (the
    // FFScrollDuringPlaybackKey NSUserDefaults toggle). If it's OFF, we
    // should only draw the smooth 120Hz overlay line and let FCP's native
    // edge-tracking handle the scroll (playhead slides right, timeline
    // stays put until the playhead hits the side threshold). If it's ON,
    // we take over the scroll so centering happens smoothly at display
    // refresh instead of FCP's 30Hz step-based centering.
    BOOL userWantsCentered = PO_scrollDuringPlayback(view);

    // Safety gate: only pause Apple's scroll machinery if we can actually
    // drive our own replacement. If locationRangeForTime: or the clip-view
    // lookup fails, leave Apple's scroller running and show only a cosmetic
    // line on top — still a visual win, no functional regression.
    BOOL canDriveScroll = NO;
    if (userWantsCentered && [view respondsToSelector:phSel]) {
        PO_CMTime probeTime = ((PO_CMTime (*)(id, SEL))PO_STRET)(view, phSel);
        double probeX = 0.0;
        BOOL gotX = (probeTime.timescale > 0) && PO_xForTime(view, probeTime, &probeX);
        BOOL gotClip = [view.superview isKindOfClass:[NSClipView class]];
        canDriveScroll = gotX && gotClip && isfinite(probeX);
        if (!canDriveScroll) {
            SpliceKit_log(@"[PlayheadOverlay] Safety gate: not pausing Apple scroller "
                          @"(gotX=%d gotClip=%d) — overlay-only mode",
                          (int)gotX, (int)gotClip);
        }
    }

    // Pause Apple's auto-scroll *only* when the user has centered-during-
    // playback on AND our safety probe succeeded. TLKScrollingTimeline
    // otherwise runs step-based `scrollPlayheadTowardMiddle` on every
    // playhead-time update (30Hz on a 30p project); on a ProMotion display
    // that reads as the timeline hopping sideways a few times per second.
    // When centered is OFF, we want Apple's edge-threshold scroll left
    // intact — the user explicitly chose "playhead slides off to the right
    // until it reaches the edge," and our 120Hz overlay already gives them
    // a smooth line.
    if (userWantsCentered && canDriveScroll) {
        @try {
            SEL stSel = NSSelectorFromString(@"scrollingTimeline");
            id scrollingTimeline = [view respondsToSelector:stSel]
                ? ((id (*)(id, SEL))objc_msgSend)(view, stSel) : nil;
            if (scrollingTimeline) {
                SEL pausedGet = NSSelectorFromString(@"paused");
                SEL pausedSet = NSSelectorFromString(@"setPaused:");
                if ([scrollingTimeline respondsToSelector:pausedGet]) {
                    sScrollingTimelineWasPaused =
                        ((BOOL (*)(id, SEL))objc_msgSend)(scrollingTimeline, pausedGet);
                }
                if ([scrollingTimeline respondsToSelector:pausedSet]) {
                    ((void (*)(id, SEL, BOOL))objc_msgSend)(scrollingTimeline, pausedSet, YES);
                    sPausedScrollingTimelineWeak = scrollingTimeline;
                    sPausedTimelineViewWeak = view;
                    SpliceKit_log(@"[PlayheadOverlay] Paused TLKScrollingTimeline (centered-playback mode)");
                }
            }
        } @catch (...) {}
    }

    // Keep the native playhead visible until the first display-link tick has
    // both a valid time-to-x mapping and an attached replacement layer.
    sOverlayLayer.hidden = YES;
    PO_startDisplayLink(view);
}

static void PO_onPlaybackEnded(void) {
    sIsPlaying = NO;
    PO_stopDisplayLink();

    PO_deactivateOverlayVisual();

    PO_restorePausedScrollingTimeline();
}

// ---- Install / remove ----

static BOOL sObserversRegistered = NO;

static void PO_registerObservers(void) {
    if (sObserversRegistered) return;
    NSNotificationCenter *nc = [NSNotificationCenter defaultCenter];

    [nc addObserverForName:@"PEPlayerDidBeginPlaybackNotification"
                    object:nil queue:[NSOperationQueue mainQueue]
                usingBlock:^(NSNotification *note) {
        if (sOverlayInstalled) PO_onPlaybackBegan();
    }];
    [nc addObserverForName:@"PEPlayerDidEndPlaybackNotification"
                    object:nil queue:[NSOperationQueue mainQueue]
                usingBlock:^(NSNotification *note) {
        if (sOverlayInstalled) PO_onPlaybackEnded();
    }];
    sObserversRegistered = YES;
}

void SpliceKit_installTimelinePlayheadOverlay(void) {
    if (sOverlayInstalled) return;

    Class tlvCls = objc_getClass("TLKTimelineView");
    if (!tlvCls) {
        SpliceKit_log(@"[PlayheadOverlay] TLKTimelineView not found — skip");
        return;
    }

    SEL sel = @selector(_setPlayheadTime_NoKVO:animate:);
    Method m = class_getInstanceMethod(tlvCls, sel);
    if (m) {
        sOrigSetPlayheadTimeNoKVO = method_setImplementation(
            m, (IMP)SpliceKit_swizzled_setPlayheadTimeNoKVO);
        SpliceKit_log(@"[PlayheadOverlay] Swizzled -[TLKTimelineView _setPlayheadTime_NoKVO:animate:]");
    } else {
        SpliceKit_log(@"[PlayheadOverlay] _setPlayheadTime_NoKVO:animate: not found — overlay will not track playback");
    }

    PO_registerObservers();
    sOverlayInstalled = YES;
    SpliceKit_log(@"[PlayheadOverlay] Installed");
}

void SpliceKit_removeTimelinePlayheadOverlay(void) {
    if (!sOverlayInstalled) return;
    PO_stopDisplayLink();

    // Restore real playhead if we hid it.
    PO_deactivateOverlayVisual();
    PO_restorePausedScrollingTimeline();

    if (sOverlayLayer) {
        [sOverlayLayer removeFromSuperlayer];
        sOverlayLayer = nil;
    }

    Class tlvCls = objc_getClass("TLKTimelineView");
    if (tlvCls && sOrigSetPlayheadTimeNoKVO) {
        Method m = class_getInstanceMethod(tlvCls, @selector(_setPlayheadTime_NoKVO:animate:));
        if (m) method_setImplementation(m, sOrigSetPlayheadTimeNoKVO);
        sOrigSetPlayheadTimeNoKVO = NULL;
    }

    sOverlayInstalled = NO;
    sIsPlaying = NO;
    SpliceKit_log(@"[PlayheadOverlay] Removed");
}

void SpliceKit_setTimelinePlayheadOverlayEnabled(BOOL enabled) {
    [[NSUserDefaults standardUserDefaults] setBool:enabled forKey:kDefTimelinePlayheadOverlay];
    if (enabled) {
        SpliceKit_installTimelinePlayheadOverlay();
    } else {
        SpliceKit_removeTimelinePlayheadOverlay();
    }
}

BOOL SpliceKit_isTimelinePlayheadOverlayEnabled(void) {
    NSNumber *n = [[NSUserDefaults standardUserDefaults]
                   objectForKey:kDefTimelinePlayheadOverlay];
    return n ? [n boolValue] : NO;
}
