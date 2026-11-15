package tmk

import (
	"math"
	"testing"
)

func hashWith(t *testing.T, periods []int, coeffs []float64, n, seed int) *Hash {
	t.Helper()
	h := NewWith(periods, coeffs)
	for i, f := range frames(n, seed) {
		h.Add(i, f)
	}
	out, err := h.Finish()
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// TestEveryWayTwoHashesCanBeIncomparable walks each arm of the guard
// separately. ⛔ A guard tested through one arm only is a guard whose other
// arms have never run, and this one stands between a caller and a number that
// would look entirely plausible.
func TestEveryWayTwoHashesCanBeIncomparable(t *testing.T) {
	base := hashWith(t, []int{101, 211}, PoullotCoefficients[:], 40, 1)

	cases := []struct {
		name    string
		periods []int
		coeffs  []float64
	}{
		{"a different number of periods", []int{101}, PoullotCoefficients[:]},
		{"the same count, different periods", []int{101, 307}, PoullotCoefficients[:]},
		{"a different number of coefficients", []int{101, 211}, PoullotCoefficients[:16]},
		{"the same count, different coefficients", []int{101, 211}, append([]float64{0.5}, PoullotCoefficients[1:]...)},
	}
	for _, c := range cases {
		other := hashWith(t, c.periods, c.coeffs, 40, 1)
		if _, err := Level1(base, other); err != ErrIncomparable {
			t.Errorf("%s: Level1 gave %v, want ErrIncomparable", c.name, err)
		}
		if _, err := Level2(base, other); err != ErrIncomparable {
			t.Errorf("%s: Level2 gave %v, want ErrIncomparable", c.name, err)
		}
	}
}

// TestAnEmptyFeatureScoresZeroRatherThanNaN: a video whose frames are all
// zero has no direction, and cosine similarity of a zero vector is undefined.
// Returning 0 keeps a NaN from propagating into a caller's threshold, where
// every comparison against it would silently be false.
func TestAnEmptyFeatureScoresZeroRatherThanNaN(t *testing.T) {
	h := NewWith([]int{101}, PoullotCoefficients[:4])
	var zero [Dim]float32
	for i := 0; i < 10; i++ {
		h.Add(i, zero)
	}
	a, err := h.Finish()
	if err != nil {
		t.Fatal(err)
	}

	got, err := Level1(a, a)
	if err != nil {
		t.Fatal(err)
	}
	if got != 0 {
		t.Fatalf("level-1 of an all-zero video = %v, want 0", got)
	}
	if math.IsNaN(got) {
		t.Fatal("level-1 returned NaN")
	}

	// The level-2 slots are all zero too, so the kernel is zero everywhere.
	l2, err := Level2(a, a)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(l2) || l2 < 0 || l2 > 1 {
		t.Fatalf("level-2 of an all-zero video = %v, want a value in [0,1]", l2)
	}
}

// TestNoPeriodsScoresZero covers the degenerate parameter set: with nothing
// to slide over there is no kernel, and the answer is 0 rather than an
// infinity escaping into a caller's comparison.
func TestNoPeriodsScoresZero(t *testing.T) {
	h := NewWith(nil, PoullotCoefficients[:])
	for i, f := range frames(20, 2) {
		h.Add(i, f)
	}
	a, err := h.Finish()
	if err != nil {
		t.Fatal(err)
	}
	got, err := Level2(a, a)
	if err != nil {
		t.Fatal(err)
	}
	if got != 0 {
		t.Fatalf("level-2 with no periods = %v, want 0", got)
	}
}

// TestScoreStaysInRange pins the clamp. Floating-point accumulation can carry
// a self-comparison a hair above its own normaliser, and a "similarity" of
// 1.0000001 is the kind of value that passes every threshold and then fails
// an assertion three layers away.
func TestScoreStaysInRange(t *testing.T) {
	h := hashAt(t, frames(500, 13), 0)
	got, err := Level2(h, h)
	if err != nil {
		t.Fatal(err)
	}
	if got < 0 || got > 1 {
		t.Fatalf("level-2 self score = %v, outside [0,1]", got)
	}
}
