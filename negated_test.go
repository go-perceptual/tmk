package tmk

import "testing"

// TestAnAntiCorrelatedPairDoesNotMatch feeds the exact negation of a video,
// which is the strongest anti-correlation that exists for this kernel.
//
// ⚠ It also records a PREDICTION THAT WAS WRONG, because the wrong answer is
// the useful part. Feeding the negation makes every level-2 slot negate too,
// giving
//
//	K(d) = -a_0 - 2 * sum_{j>=1} cos(j d) a_j
//
// and I argued the maximum over d must stay negative, since no d makes
// cos(j d) = -1 for every j at once. Measured, it is **+0.037**: the
// oscillating part does climb past a_0. The score is small — nowhere near
// matching — but it is not negative.
//
// That is why Level2 clamps with Min/Max rather than two `if`s: there is no
// known input that drives the score below zero, and a branch nothing can
// reach is a branch nobody has checked.
func TestAnAntiCorrelatedPairDoesNotMatch(t *testing.T) {
	fs := frames(200, 23)

	a := hashAt(t, fs, 0)

	neg := make([][Dim]float32, len(fs))
	for i, f := range fs {
		var n [Dim]float32
		for k, v := range f {
			n[k] = -v
		}
		neg[i] = n
	}
	b := hashAt(t, neg, 0)

	got, err := Level2(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if got < 0 || got > 1 {
		t.Fatalf("an anti-correlated pair scored %v, outside [0,1]", got)
	}
	if got >= DefaultThreshold {
		t.Fatalf("an anti-correlated pair scored %v, at or above the %v threshold", got, DefaultThreshold)
	}

	// And the control: level-1 sees the anti-correlation as -1, which is the
	// honest answer there, so the clamp belongs to level-2 alone.
	l1, err := Level1(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if l1 > -0.99 {
		t.Fatalf("level-1 of a negated video = %v, want about -1", l1)
	}
}
