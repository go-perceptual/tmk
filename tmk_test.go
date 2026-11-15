package tmk

import (
	"math"
	"math/rand"
	"testing"
)

// frames makes a deterministic pseudo-video: n frame features that drift
// smoothly, the way consecutive frames of real footage do, with a seed that
// makes two "videos" unrelated.
func frames(n, seed int) [][Dim]float32 {
	rng := rand.New(rand.NewSource(int64(seed)))
	var cur [Dim]float32
	for k := range cur {
		cur[k] = float32(rng.NormFloat64())
	}
	out := make([][Dim]float32, n)
	for i := range out {
		for k := range cur {
			cur[k] += float32(rng.NormFloat64() * 0.05) // slow drift
		}
		out[i] = cur
	}
	return out
}

func hashAt(t *testing.T, fs [][Dim]float32, offset int) *Hash {
	t.Helper()
	h := New()
	for i, f := range fs {
		h.Add(offset+i, f)
	}
	out, err := h.Finish()
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// TestAHashScoresOneAgainstItself is the identity the normaliser exists for:
// it is a_0 + 2*sum a_j precisely so that a self-comparison lands on 1.0. If
// the normalisation of the features and this closed form ever disagree, every
// score silently changes scale and nothing else would notice.
func TestAHashScoresOneAgainstItself(t *testing.T) {
	h := hashAt(t, frames(300, 1), 0)

	l1, err := Level1(h, h)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(l1-1) > 1e-6 {
		t.Errorf("level-1 self score = %v, want 1", l1)
	}

	l2, err := Level2(h, h)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(l2-1) > 1e-6 {
		t.Errorf("level-2 self score = %v, want 1", l2)
	}
}

// TestUnrelatedVideosScoreLow is the control: a scorer that returned 1 for
// everything would pass the test above.
func TestUnrelatedVideosScoreLow(t *testing.T) {
	a := hashAt(t, frames(300, 1), 0)
	b := hashAt(t, frames(300, 2), 0)

	l2, err := Level2(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if l2 >= DefaultThreshold {
		t.Fatalf("unrelated videos scored %v, at or above the %v threshold", l2, DefaultThreshold)
	}
}

// TestTheSameVideoShiftedInTimeStillMatches is what the temporal kernel is
// FOR. The same frames, starting 97 frames later, are a different sequence in
// every element-wise sense; only sliding one against the other recovers them.
//
// ⛔ This is also the test that fails if `t` is treated as a counter rather
// than a timestamp, which is the easy mistake when ingesting frames.
func TestTheSameVideoShiftedInTimeStillMatches(t *testing.T) {
	fs := frames(300, 7)
	a := hashAt(t, fs, 0)
	b := hashAt(t, fs, 97)

	l2, err := Level2(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if l2 < DefaultThreshold {
		t.Fatalf("the same video shifted by 97 frames scored %v, below the %v threshold", l2, DefaultThreshold)
	}
}

// TestLevel1IgnoresOrder demonstrates, rather than asserts in prose, why
// level-1 is only a filter: it is the mean frame, so the same frames in a
// different order give the same answer.
func TestLevel1IgnoresOrder(t *testing.T) {
	fs := frames(200, 3)
	forward := hashAt(t, fs, 0)

	rev := make([][Dim]float32, len(fs))
	for i := range fs {
		rev[i] = fs[len(fs)-1-i]
	}
	backward := hashAt(t, rev, 0)

	l1, err := Level1(forward, backward)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(l1-1) > 1e-6 {
		t.Fatalf("level-1 of a reversed video = %v, want 1: it is the mean frame and must ignore order", l1)
	}
}

func TestNoFramesIsAnError(t *testing.T) {
	if _, err := New().Finish(); err != ErrNoFrames {
		t.Fatalf("finishing an empty hasher gave %v, want ErrNoFrames", err)
	}
}

// TestMismatchedParametersAreRefused: a score across different periods is
// meaningless and looks exactly like a real one, so it is an error.
func TestMismatchedParametersAreRefused(t *testing.T) {
	a := hashAt(t, frames(50, 1), 0)

	other := NewWith([]int{101, 211}, PoullotCoefficients[:])
	for i, f := range frames(50, 1) {
		other.Add(i, f)
	}
	b, err := other.Finish()
	if err != nil {
		t.Fatal(err)
	}

	if _, err := Level1(a, b); err != ErrIncomparable {
		t.Errorf("Level1 across different periods gave %v, want ErrIncomparable", err)
	}
	if _, err := Level2(a, b); err != ErrIncomparable {
		t.Errorf("Level2 across different periods gave %v, want ErrIncomparable", err)
	}
}

// TestTheHashIsTheSameSizeWhateverTheLength is TMK's headline property: a
// ten-second clip and a two-hour film cost the same.
func TestTheHashIsTheSameSizeWhateverTheLength(t *testing.T) {
	short := hashAt(t, frames(30, 5), 0)
	long := hashAt(t, frames(3000, 5), 0)

	if short.Frames() == long.Frames() {
		t.Fatal("the corpus is wrong: both have the same frame count")
	}
	if len(short.Cos) != len(long.Cos) || len(short.Cos[0]) != len(long.Cos[0]) {
		t.Fatal("the level-2 feature changed shape with the video length")
	}
	if len(short.Cos[0][0]) != Dim || len(long.Cos[0][0]) != Dim {
		t.Fatal("a feature slot is not Dim long")
	}
}

// TestASlotHasTheNormItsCoefficientPromises pins the identity the whole
// normalisation rests on: each slot is scaled so that its squared norm is
// exactly its Fourier coefficient. The closed-form normaliser is derived from
// that, so if it drifts, every score changes scale quietly.
func TestASlotHasTheNormItsCoefficientPromises(t *testing.T) {
	h := hashAt(t, frames(400, 11), 0)
	for i := range h.Cos {
		for j := range h.Cos[i] {
			var n float64
			for _, v := range h.Cos[i][j] {
				n += float64(v) * float64(v)
			}
			want := PoullotCoefficients[j]
			if n == 0 {
				continue // an all-zero slot is left alone on purpose
			}
			if math.Abs(n-want) > want*1e-5 {
				t.Fatalf("cos[%d][%d] has squared norm %v, want %v", i, j, n, want)
			}
		}
	}
}

func TestNormalizerIsTheClosedForm(t *testing.T) {
	got := normalizerOf([]float64{0.5, 0.25, 0.125})
	want := 0.5 + 2*0.25 + 2*0.125
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("normalizerOf = %v, want %v", got, want)
	}
	if normalizerOf(nil) != 1 {
		t.Fatal("an empty coefficient set must not divide by zero")
	}
	if normalizerOf([]float64{0}) != 1 {
		t.Fatal("an all-zero coefficient set must not divide by zero")
	}
}
