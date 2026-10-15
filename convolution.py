"""Convolution, step by step — a code companion to docs/convolution.txt.

Run:  uv run convolution.py
Pure Python (no dependencies) so every multiply and add is visible.
"""

from fractions import Fraction


def header(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def bar_plot(xs, ys, width=40):
    top = max(abs(y) for y in ys) or 1
    for x, y in zip(xs, ys):
        print(f"  {x:6.2f} | {'#' * round(width * y / top):<{width}} {y:.3f}")


def rect(t):
    """Rectangular pulse: 1 on [0, 1], 0 elsewhere."""
    return 1.0 if 0 <= t <= 1 else 0.0


def integrate(func, a, b, n=4000):
    """Midpoint Riemann sum: adds up func over [a, b]."""
    dx = (b - a) / n
    return sum(func(a + (i + 0.5) * dx) for i in range(n)) * dx


# ---------------------------------------------------------------------------
# STEP 1: Multiply point by point, then integrate = "total overlap"
# ---------------------------------------------------------------------------
def step1_multiply_and_integrate():
    header("STEP 1: multiply point-by-point, then integrate")
    print("f = rect pulse on [0,1]; g = same pulse shifted right by s.")
    print("The product f*g is 1 only where BOTH are 1, so its integral = overlap length.\n")
    ts = [i / 10 for i in range(-5, 26)]

    def strip(func):
        return "".join("#" if func(t) else "." for t in ts)

    print(f"  t from {ts[0]} to {ts[-1]} in steps of 0.1  ('#' = 1, '.' = 0)\n")
    print(f"  f(t)          {strip(rect)}")
    for s in [0.0, 0.25, 0.5, 1.0, 1.5]:
        product = lambda t: rect(t) * rect(t - s)
        overlap = integrate(product, -1, 3)
        print(f"\n  s = {s:4.2f}  g(t) {strip(lambda t: rect(t - s))}")
        print(f"       f(t)*g(t) {strip(product)}   integral = {overlap:.3f}")


# ---------------------------------------------------------------------------
# STEP 2: Slide one function across the other -> a new function of t
#   (f * g)(t) = ∫ f(τ) g(t - τ) dτ
# ---------------------------------------------------------------------------
def continuous_convolve(f, g, t, lo=-3, hi=4):
    return integrate(lambda tau: f(tau) * g(t - tau), lo, hi)


def exact_triangle(t):
    if 0 <= t <= 1:
        return t
    if 1 < t <= 2:
        return 2 - t
    return 0.0


def step2_continuous_convolution():
    header("STEP 2: slide, and record the overlap at every t  ->  rect * rect")
    print("(f*g)(t) = ∫ f(τ) g(t-τ) dτ.  Numeric result vs. exact triangle:\n")
    ts = [i / 4 for i in range(-2, 11)]
    ys = [continuous_convolve(rect, rect, t) for t in ts]
    for t, y in zip(ts, ys):
        print(f"  t = {t:5.2f}   numeric = {y:.3f}   exact = {exact_triangle(t):.3f}")
    print("\nShape of the output (two rectangles make a triangle, width 1+1 = 2):")
    bar_plot(ts, ys)


# ---------------------------------------------------------------------------
# STEP 3: Discrete convolution — the same idea with sums instead of integrals
#   (f * g)[n] = Σ_k f[k] g[n - k]
# ---------------------------------------------------------------------------
def convolve(f, g):
    """Full discrete convolution straight from the definition."""
    out = [0] * (len(f) + len(g) - 1)
    for n in range(len(out)):
        for k in range(len(f)):
            if 0 <= n - k < len(g):
                out[n] += f[k] * g[n - k]
    return out


def correlate(f, g):
    """Same sliding, but WITHOUT flipping g (what most CNN libraries do)."""
    return convolve(f, g[::-1])


def step3_discrete_convolution():
    header("STEP 3: discrete convolution  [1,2,1] * [1,1]")
    f, g = [1, 2, 1], [1, 1]
    for n in range(len(f) + len(g) - 1):
        terms = [f"f[{k}]*g[{n - k}]" for k in range(len(f)) if 0 <= n - k < len(g)]
        values = [f"{f[k]}*{g[n - k]}" for k in range(len(f)) if 0 <= n - k < len(g)]
        print(f"  out[{n}] = {' + '.join(terms):<22} = {' + '.join(values):<10} = {convolve(f, g)[n]}")
    print(f"\n  result = {convolve(f, g)}   length = {len(f)} + {len(g)} - 1 = {len(f) + len(g) - 1}")


# ---------------------------------------------------------------------------
# STEP 4: Why flip?  Feed in a single impulse and see what comes out.
# ---------------------------------------------------------------------------
def step4_why_flip():
    header("STEP 4: why the flip?  (asymmetric kernel)")
    impulse = [0, 0, 1, 0, 0]
    kernel = [1, 2, 3]
    print(f"  input  = {impulse}  (a single spike)")
    print(f"  kernel = {kernel}   (asymmetric)\n")
    print(f"  convolution (flip)    -> {convolve(impulse, kernel)}")
    print(f"  correlation (no flip) -> {correlate(impulse, kernel)}")
    print("\n  With the flip, a spike reproduces the kernel in its natural order.")
    print("  Without it, the kernel comes out backwards. For symmetric kernels")
    print(f"  like [1,2,1] both agree: {convolve(impulse, [1, 2, 1])} == {correlate(impulse, [1, 2, 1])}")


# ---------------------------------------------------------------------------
# STEP 5: Systems — output = input * impulse response  (a room echo)
# ---------------------------------------------------------------------------
def step5_echo():
    header("STEP 5: impulse response — clapping in a room")
    room = [1.0, 0, 0, 0.5, 0, 0, 0.25]  # direct sound, then two fading echoes
    one_clap = [1.0]
    two_claps = [1.0, 0, 0.6]  # a loud clap, then a softer one
    print(f"  room impulse response h = {room}")
    print(f"  one clap  * h = {convolve(one_clap, room)}")
    print(f"  two claps * h = {[round(v, 2) for v in convolve(two_claps, room)]}")
    print("\n  The second output is a sum of two scaled, shifted copies of h:")
    a = [1.0 * v for v in room] + [0, 0]
    b = [0, 0] + [0.6 * v for v in room]
    print(f"     1.0 * h           = {[round(v, 2) for v in a]}")
    print(f"     0.6 * h shifted 2 = {[round(v, 2) for v in b]}")
    print(f"     sum               = {[round(x + y, 2) for x, y in zip(a, b)]}")


# ---------------------------------------------------------------------------
# STEP 6: 2-D convolution on an "image" — blur and edge detection
# ---------------------------------------------------------------------------
def filter2d(image, kernel):
    """'Valid' 2-D sliding window, no flip (cross-correlation, as in CNN libraries)."""
    kh, kw = len(kernel), len(kernel[0])
    out = []
    for i in range(len(image) - kh + 1):
        row = []
        for j in range(len(image[0]) - kw + 1):
            row.append(sum(image[i + a][j + b] * kernel[a][b] for a in range(kh) for b in range(kw)))
        out.append(row)
    return out


def print_grid(grid):
    for row in grid:
        print("   " + " ".join(f"{v:5.1f}" for v in row))


def step6_images():
    header("STEP 6: images — a 3x3 kernel slides over a pixel grid")
    image = [[0, 0, 0, 9, 9, 9] for _ in range(6)]  # dark left half, bright right half
    print("  image (vertical edge between columns 2 and 3):")
    print_grid(image)

    blur = [[1 / 9] * 3 for _ in range(3)]
    print("\n  box blur (every weight 1/9) -> the sharp jump becomes a gradual ramp:")
    print_grid(filter2d(image, blur))

    edge = [[-1, 0, 1]] * 3  # right minus left
    print("\n  vertical-edge kernel [-1 0 1] -> large only where brightness changes:")
    print_grid(filter2d(image, edge))


# ---------------------------------------------------------------------------
# STEP 7: Probability — distribution of the sum of two dice
# ---------------------------------------------------------------------------
def step7_dice():
    header("STEP 7: probability — sum of two dice = convolution of their PMFs")
    die = [Fraction(0)] + [Fraction(1, 6)] * 6  # index = face value 0..6
    total = convolve(die, die)  # index = sum 0..12
    for s in range(2, 13):
        print(f"  P(sum = {s:2}) = {str(total[s]):>5}  {'#' * (total[s] * 36).numerator}")
    print(f"\n  P(sum = 7) = {total[7]}  (= 6/36: the six pairs 1+6, 2+5, ..., 6+1)")


def main():
    step1_multiply_and_integrate()
    step2_continuous_convolution()
    step3_discrete_convolution()
    step4_why_flip()
    step5_echo()
    step6_images()
    step7_dice()
    print("\nSummary: flip, shift, multiply, integrate (or sum).")


if __name__ == "__main__":
    main()
