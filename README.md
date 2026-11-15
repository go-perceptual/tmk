# tmk

TMK+PDQF, a video-similarity hash, in pure Go with no dependencies and no cgo.

```go
h := tmk.New()
for t, feature := range features {   // one [256]float32 per frame, at 15 fps
    h.Add(t, feature)
}
hash, err := h.Finish()

s, _ := tmk.Level1(a, b)   // cheap filter: cosine of the mean frame
s, _ = tmk.Level2(a, b)    // the real score, 1.0 identical, 0.0 worst
```

A score of **0.7 or more**, at either level, is the reference's validated
threshold.

## What it is for

TMK answers "are these the same video?" when the two differ in frame rate,
container, resolution, light watermarking, or a few seconds of leader and
trailer. It is **not** built to find a clip inside a longer video, nor to
survive cropping or heavy overlays.

⭐ Its useful property is that the signature is a **fixed size whatever the
video's length** — about 256 KB, of which the first kilobyte already separates
almost every pair. A library of videos costs a predictable amount of memory
and can be compared without the originals.

## Two levels, and why there are two

**Level 1** is the cosine similarity of the mean frame feature. It carries no
timing information at all — the same shots in a different order score the
same, and a test demonstrates exactly that by hashing a video backwards. It is
a filter, cheap enough to index.

**Level 2** is the temporal kernel. For each period it slides one video
against the other over every whole-frame offset and keeps the best match, then
takes the best period. That sliding is what matches a video against itself
with the leader removed: a test shifts a video by 97 frames and it still
scores above threshold.

The reference computes level 2 only when level 1 passes, because when level-1
scores differ greatly the level-2 scores do too.

## ⛔ It does not decode video

This package takes **frame features**, one `[256]float32` per frame — which is
what [go-perceptual/pdq](https://github.com/go-perceptual/pdq)'s `Float`
returns. TMK+PDQF is defined as PDQ minus its binarisation step, so the
descriptor is the image hash's own float stage.

Decoding belongs to whoever owns the container and the codec. Keeping it out
means this package is testable without a single video file, and usable with
any decoder you already have.

⛔ `Add` takes the frame's **timestamp**, not a counter. The kernel's entire
content is where a frame sits in time, so a caller that drops frames must keep
the original indices rather than renumbering what it kept.

⛔ Frames must be resampled to **15 fps** before hashing. That is part of the
algorithm, not a tuning knob: the kernel compares phases at integer frame
indices, so two videos hashed at different rates are not comparable even
though both hashes look perfectly valid.

## Parameters

The four periods `{2731, 4391, 9767, 14653}` and the 32 Fourier coefficients
are the Poullot values used by the reference. `NewWith` takes your own, and
`Level1`/`Level2` **refuse** a pair built with different ones: a score across
mismatched periods is meaningless and looks exactly like a real one.

## Where it comes from

Published as *The TMK+PDQF video-hashing algorithm and the PDQ image-hashing
algorithm* (`hashing/hashing.pdf` in
[facebook/ThreatExchange](https://github.com/facebook/ThreatExchange)), with
the kernel from Poullot et al. Ported from the reference C++ in `tmk/cpp` and
from the derivation in its `tmkfv.h`, both read rather than summarised.

## Licence

BSD-3-Clause.
