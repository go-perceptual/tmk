package tmk

import (
	"errors"
	"math"
)

// ErrIncomparable is returned when two hashes were built with different
// parameters. ⛔ It is an error rather than a number because a score computed
// across mismatched periods is meaningless and looks exactly like a real one.
var ErrIncomparable = errors.New("tmk: hashes were built with different parameters")

func comparable(a, b *Hash) error {
	if len(a.periods) != len(b.periods) || len(a.coefficients) != len(b.coefficients) {
		return ErrIncomparable
	}
	for i := range a.periods {
		if a.periods[i] != b.periods[i] {
			return ErrIncomparable
		}
	}
	for i := range a.coefficients {
		if a.coefficients[i] != b.coefficients[i] {
			return ErrIncomparable
		}
	}
	return nil
}

// Level1 is the cosine similarity of the two level-1 features: 1.0 for the
// same video, -1.0 at the opposite end.
//
// It carries NO timing information — it is the mean frame, and two videos
// with the same shots in a different order score the same. That is exactly
// why it is only a filter: the reference computes the expensive level-2
// score only when this one passes, because when level-1 scores differ
// greatly the level-2 scores do too.
func Level1(a, b *Hash) (float64, error) {
	if err := comparable(a, b); err != nil {
		return 0, err
	}
	var dot, na, nb float64
	for k := 0; k < Dim; k++ {
		x, y := float64(a.Average[k]), float64(b.Average[k])
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 0, nil
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb)), nil
}

// Level2 is the TMK pair score: 1.0 for identical hashes, 0.0 at worst.
//
// For each period it slides one video against the other over every whole-frame
// offset and keeps the best match, then takes the best period. That sliding is
// what lets it match a video against the same video with a few seconds of
// leader removed.
//
// The kernel at offset delta is
//
//	K = u_c0 . v_c0
//	  + sum_j cos(j d) (u_cj . v_cj + u_sj . v_sj)
//	  + sum_j sin(j d) (u_sj . v_cj - u_cj . v_sj)
//
// ⭐ The four dot products do not depend on the offset, so they are computed
// ONCE per coefficient and then combined for each of the thousands of
// offsets. Recomputing them inside the offset loop — the obvious reading of
// the formula — multiplies the work by 256 and changes no answer.
func Level2(a, b *Hash) (float64, error) {
	if err := comparable(a, b); err != nil {
		return 0, err
	}
	m := len(a.coefficients)
	best := math.Inf(-1)

	for i, T := range a.periods {
		// cc[j] = u_cj . v_cj, and so on. Offset-independent.
		cc := make([]float64, m)
		ss := make([]float64, m)
		sc := make([]float64, m)
		cs := make([]float64, m)
		for j := 0; j < m; j++ {
			cc[j] = dot(a.Cos[i][j], b.Cos[i][j])
			ss[j] = dot(a.Sin[i][j], b.Sin[i][j])
			sc[j] = dot(a.Sin[i][j], b.Cos[i][j])
			cs[j] = dot(a.Cos[i][j], b.Sin[i][j])
		}

		for offset := 0; offset < T; offset++ {
			delta := 2 * math.Pi * float64(offset) / float64(T)
			k := cc[0]
			for j := 1; j < m; j++ {
				jd := float64(j) * delta
				k += math.Cos(jd)*(cc[j]+ss[j]) + math.Sin(jd)*(sc[j]-cs[j])
			}
			if k > best {
				best = k
			}
		}
	}

	if math.IsInf(best, -1) {
		return 0, nil // no periods: nothing to score
	}
	// Clamped to [0,1] because the published score is a similarity, and
	// floating-point accumulation can carry a self-comparison a hair past its
	// own normaliser. A score of 1.0000001 passes every threshold and fails
	// an assertion three layers away.
	//
	// ⛔ Written with Min/Max rather than two `if`s on purpose. As branches,
	// the lower one has no input known to reach it: an exactly anti-
	// correlated pair — the obvious candidate — measures +0.037, not
	// negative, because the oscillating part of the kernel climbs past a_0.
	// Two attempts to reason about reachability gave opposite answers and the
	// measurement refuted the second, so the guard stays and the UNTESTABLE
	// BRANCH goes: this form has the same behaviour and nothing uncoverable.
	return math.Max(0, math.Min(1, best/a.normalizer)), nil
}

func dot(x, y []float32) float64 {
	var s float64
	for k := range x {
		s += float64(x[k]) * float64(y[k])
	}
	return s
}
