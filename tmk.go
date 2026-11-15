// Package tmk implements TMK+PDQF, a video-similarity hash.
//
// TMK answers "are these two videos the same video?" for videos that differ
// in frame rate, container, resolution, light watermarking or a few seconds
// of leader and trailer. It is NOT built to find a clip inside a longer
// video, nor to survive cropping or heavy overlays.
//
// Its useful property is that the signature has a FIXED SIZE whatever the
// video's length — about 256 KB, of which the first kilobyte already
// separates almost every pair — so a library of videos costs a predictable
// amount of memory and can be compared without the originals.
//
// The algorithm is published as "The TMK+PDQF video-hashing algorithm and
// the PDQ image-hashing algorithm" (hashing/hashing.pdf in
// github.com/facebook/ThreatExchange), with the kernel from Poullot et al.
// This package is a port of the reference C++ in tmk/cpp of that repository,
// and of the derivation written in its tmkfv.h.
//
// ⛔ This package does NOT decode video. It takes frame features — one
// [256]float32 per frame, which is what go-perceptual/pdq's Float returns —
// at a fixed frame rate. Decoding belongs to whoever owns the container and
// the codec, and keeping it out means this package is testable without a
// single video file and usable with any decoder.
package tmk

import (
	"errors"
	"math"
)

// FPS is the frame rate frames must be resampled to before hashing.
//
// ⛔ It is part of the algorithm, not a tuning knob: the temporal kernel
// compares phases at integer frame indices, so two videos hashed at
// different rates are not comparable even though both hashes look valid.
const FPS = 15

// Dim is the length of one frame feature: PDQF, the PDQ transform before its
// binarisation step.
const Dim = 256

// PoullotPeriods are the four periods of the temporal kernel, from Poullot
// et al. via the reference implementation.
var PoullotPeriods = [...]int{2731, 4391, 9767, 14653}

// PoullotCoefficients are the 32 Fourier weights, likewise.
var PoullotCoefficients = [...]float64{
	0.0708041893112, 0.13937789309, 0.132897260304,
	0.122765735552, 0.109878684888, 0.09529606433,
	0.0800986647852, 0.0652590650356, 0.0515478238322,
	0.0394851531195, 0.0293374252025, 0.0211492623679,
	0.0147973073245, 0.0100512818746, 0.0066306408014,
	0.00424947117334, 0.0026467615764, 0.00160270959695,
	0.000943882629639, 0.000540841638603, 0.000301633183798,
	0.000163800158855, 8.66454753015e-05, 4.46626303151e-05,
	2.24429442235e-05, 1.09982139799e-05, 5.25823487999e-06,
	2.45358229988e-06, 1.11781474895e-06, 4.97406489221e-07,
	2.16265487234e-07, 9.19087006565e-08,
}

// DefaultThreshold is the reference's validated threshold for both the
// level-1 and the level-2 score.
const DefaultThreshold = 0.7

// ErrNoFrames is returned when a hash is finished without any frame.
var ErrNoFrames = errors.New("tmk: no frames were ingested")

// Hash is one video's signature.
type Hash struct {
	// Average is the level-1 feature: the plain mean of the frame features,
	// carrying no timing information at all. Comparing two of these is cheap
	// and is what a large index stores.
	Average [Dim]float32
	// Cos and Sin are the level-2 features, indexed [period][coefficient][k].
	Cos [][][]float32
	Sin [][][]float32

	periods      []int
	coefficients []float64
	// normalizer makes a hash score exactly 1 against itself.
	normalizer float64
	frames     int
}

// Hasher accumulates frame features into a Hash.
type Hasher struct {
	periods      []int
	coefficients []float64
	average      [Dim]float64
	cos          [][][]float64
	sin          [][][]float64
	frames       int
}

// New returns a Hasher using the Poullot parameters.
func New() *Hasher {
	return NewWith(PoullotPeriods[:], PoullotCoefficients[:])
}

// NewWith returns a Hasher with explicit parameters. ⛔ Two hashes are only
// comparable when they were built with the SAME periods and coefficients;
// Level1 and Level2 refuse a mismatched pair rather than returning a number
// that means nothing.
func NewWith(periods []int, coefficients []float64) *Hasher {
	h := &Hasher{
		periods:      append([]int(nil), periods...),
		coefficients: append([]float64(nil), coefficients...),
	}
	h.cos = alloc3(len(periods), len(coefficients))
	h.sin = alloc3(len(periods), len(coefficients))
	return h
}

func alloc3(p, c int) [][][]float64 {
	out := make([][][]float64, p)
	for i := range out {
		out[i] = make([][]float64, c)
		for j := range out[i] {
			out[i][j] = make([]float64, Dim)
		}
	}
	return out
}

// Add ingests the feature of the frame at integer time t, counted in frames
// at FPS.
//
// ⛔ t is the TIMESTAMP, not a counter: the kernel's whole content is where a
// frame sits in time, so a caller that drops frames must keep the original
// indices rather than renumbering what it kept.
func (h *Hasher) Add(t int, feature [Dim]float32) {
	// The level-1 feature takes the frame as it is...
	for k := 0; k < Dim; k++ {
		h.average[k] += float64(feature[k])
	}

	// ...while the level-2 features take it L2-normalised, so that a bright
	// frame does not count for more than a dim one in the temporal kernel.
	var norm float64
	for k := 0; k < Dim; k++ {
		norm += float64(feature[k]) * float64(feature[k])
	}
	norm = math.Sqrt(norm)
	var unit [Dim]float64
	if norm > 0 {
		for k := 0; k < Dim; k++ {
			unit[k] = float64(feature[k]) / norm
		}
	}

	for i, T := range h.periods {
		// j = 0 is the plain sum: cos(0) is 1 and the sine term is zero, so
		// the reference adds the feature directly and leaves sin[i][0] at
		// zero for ever.
		for k := 0; k < Dim; k++ {
			h.cos[i][0][k] += unit[k]
		}
		for j := 1; j < len(h.coefficients); j++ {
			arg := 2 * math.Pi * float64(j) * float64(t) / float64(T)
			c, s := math.Cos(arg), math.Sin(arg)
			for k := 0; k < Dim; k++ {
				h.cos[i][j][k] += unit[k] * c
				h.sin[i][j][k] += unit[k] * s
			}
		}
	}
	h.frames++
}

// Finish normalises the accumulated features and returns the hash.
func (h *Hasher) Finish() (*Hash, error) {
	if h.frames == 0 {
		return nil, ErrNoFrames
	}
	out := &Hash{
		periods:      h.periods,
		coefficients: h.coefficients,
		frames:       h.frames,
		Cos:          alloc3f32(len(h.periods), len(h.coefficients)),
		Sin:          alloc3f32(len(h.periods), len(h.coefficients)),
	}
	for k := 0; k < Dim; k++ {
		out.Average[k] = float32(h.average[k] / float64(h.frames))
	}
	for i := range h.periods {
		for j := range h.coefficients {
			// Each slot is L2-normalised and then scaled by sqrt(a_j), so
			// its squared norm is exactly a_j. That identity is what makes
			// the self-score normaliser below a closed form rather than a
			// measurement.
			scale := math.Sqrt(h.coefficients[j])
			copyScaled(out.Cos[i][j], h.cos[i][j], scale)
			copyScaled(out.Sin[i][j], h.sin[i][j], scale)
		}
	}
	out.normalizer = normalizerOf(h.coefficients)
	return out, nil
}

func alloc3f32(p, c int) [][][]float32 {
	out := make([][][]float32, p)
	for i := range out {
		out[i] = make([][]float32, c)
		for j := range out[i] {
			out[i][j] = make([]float32, Dim)
		}
	}
	return out
}

func copyScaled(dst []float32, src []float64, scale float64) {
	var norm float64
	for _, v := range src {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		return // an all-zero slot stays zero; scaling it would divide by zero
	}
	for k := range src {
		dst[k] = float32(src[k] / norm * scale)
	}
}

// normalizerOf is a_0 + 2*sum_{j>=1} a_j, which is K at offset 0 when a hash
// is scored against itself. Dividing by it makes a self-comparison exactly 1.
func normalizerOf(a []float64) float64 {
	if len(a) == 0 {
		return 1
	}
	n := a[0]
	for _, v := range a[1:] {
		n += 2 * v
	}
	if n == 0 {
		return 1
	}
	return n
}

// Frames is the number of frames that went into the hash.
func (h *Hash) Frames() int { return h.frames }
