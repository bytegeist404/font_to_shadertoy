# font_to_shadertoy.py
#
# This script exports glyph contours from a TTF or OTF font file and
# formats them as GLSL arrays for use in Shadertoy shaders.
#
# The output consists of the following data structures:
# - vec2 allPoints[] = vec2[]( ... );
# - const PolyInfo allPolys[] = PolyInfo[]( PolyInfo(start, count, isHole), ... );
# - const CharInfo charTable[] = CharInfo[]( CharInfo(start, count), ... );
#
# These arrays make it possible to render font characters directly within a
# fragment shader.
#
# Installation:
# Run 'pip install -r requirements.txt' to install the necessary dependencies.
#
# Usage:
#     python font_to_shadertoy.py <path/to/font.ttf> --out <output_file.glsl>
#
# Example:
#     python font_to_shadertoy.py Roboto-Regular.ttf --out glyphs.glsl
#
# By default, the uppercase letters A-Z are exported. You can customize this
# with the '--chars' argument, for example, '--chars "Hello World"'.

import argparse
import math
import numpy as np
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen


# --- Utility Functions ---

def lerp(a, b, t):
    """Linear interpolation between a and b by factor t."""
    return a + (b - a) * t


def sample_quadratic_bezier(p0, p1, p2, n_points=12):
    """
    Generates a list of n+1 points that sample a quadratic Bézier curve
    from p0 to p2 with control point p1.
    """
    points = []
    for i in range(n_points + 1):
        t = i / n_points
        a_x = lerp(p0[0], p1[0], t)
        a_y = lerp(p0[1], p1[1], t)
        b_x = lerp(p1[0], p2[0], t)
        b_y = lerp(p1[1], p2[1], t)
        x = lerp(a_x, b_x, t)
        y = lerp(a_y, b_y, t)
        points.append((x, y))
    return points


def sample_cubic_bezier(p0, p1, p2, p3, n_points=12):
    """
    Generates a list of n+1 points that sample a cubic Bézier curve
    from p0 to p3 with control points p1 and p2.
    """
    points = []
    for i in range(n_points + 1):
        t = i / n_points
        a = (lerp(p0[0], p1[0], t), lerp(p0[1], p1[1], t))
        b = (lerp(p1[0], p2[0], t), lerp(p1[1], p2[1], t))
        c = (lerp(p2[0], p3[0], t), lerp(p2[1], p3[1], t))
        d = (lerp(a[0], b[0], t), lerp(a[1], b[1], t))
        e = (lerp(b[0], c[0], t), lerp(b[1], c[1], t))
        x = lerp(d[0], e[0], t)
        y = lerp(d[1], e[1], t)
        points.append((x, y))
    return points


def signed_area(points):
    """Calculates the signed area of a polygon."""
    area = 0.0
    num_points = len(points)
    for i in range(num_points):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % num_points]
        area += x0 * y1 - x1 * y0
    return area * 0.5


# --- Pen Class for Recording Contours ---

class RecordingPen(BasePen):
    """Collects the points of the glyph contours."""
    def __init__(self, glyph_set):
        super().__init__(glyph_set)
        self.contours = []
        self.current_contour = []

    def _moveTo(self, p0):
        if self.current_contour:
            self.contours.append(self.current_contour)
        self.current_contour = [tuple(p0)]

    def _lineTo(self, p1):
        self.current_contour.append(tuple(p1))

    def _qCurveToOne(self, p1, p2):
        p0 = self.current_contour[-1]
        points = sample_quadratic_bezier(p0, tuple(p1), tuple(p2))
        self.current_contour.extend(points[1:])

    def _curveToOne(self, p1, p2, p3):
        p0 = self.current_contour[-1]
        points = sample_cubic_bezier(p0, tuple(p1), tuple(p2), tuple(p3))
        self.current_contour.extend(points[1:])

    def _closePath(self):
        if self.current_contour:
            self.contours.append(self.current_contour)
            self.current_contour = []

    def _endPath(self):
        if self.current_contour:
            self.contours.append(self.current_contour)
            self.current_contour = []


# --- Main Extractor ---

def resample_polygon(points, n_points):
    """
    Resamples a polygon to a fixed number of points that are
    evenly distributed along its perimeter.
    """
    if len(points) < 3 or n_points < 3:
        return points

    distances = [0.0]
    for i in range(len(points)):
        a = np.array(points[i])
        b = np.array(points[(i + 1) % len(points)])
        distances.append(distances[-1] + np.linalg.norm(b - a))

    perimeter = distances[-1]
    step = perimeter / n_points
    new_points = []
    current_index = 0

    for i in range(n_points):
        target_dist = i * step
        while (
            current_index < len(distances) - 2 and distances[current_index + 1] < target_dist
        ):
            current_index += 1

        d0 = distances[current_index]
        d1 = distances[current_index + 1]
        t = (target_dist - d0) / (d1 - d0 + 1e-12)

        p0 = np.array(points[current_index % len(points)])
        p1 = np.array(points[(current_index + 1) % len(points)])

        new_p = (1 - t) * p0 + t * p1
        new_points.append(tuple(new_p))

    return new_points


def extract_glyph_contours(
    font_path,
    chars="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    target_height=1.0,
    target_points_per_contour=12,
):
    """
    Extracts contours from a font and prepares them for GLSL.

    Returns:
      - all_points: List of (x,y) tuples (flattened)
      - all_polys: List of (start_index, count, is_hole)
      - char_table: List of (start_poly_index, poly_count)
    """
    try:
        font = TTFont(font_path)
    except Exception as e:
        print(f"Error loading font '{font_path}': {e}")
        return [], [], [], chars

    glyph_set = font.getGlyphSet()
    cmap = font["cmap"].getBestCmap()
    units_per_em = font["head"].unitsPerEm

    all_points = []
    all_polys = []  # Entries: (start_point_idx, count, is_hole)
    char_table = []

    scale = target_height / units_per_em

    for ch in chars:
        code = ord(ch)
        if code not in cmap:
            char_table.append((-1, 0))
            continue
        
        glyph_name = cmap[code]
        glyph = glyph_set[glyph_name]

        pen = RecordingPen(glyph_set)
        glyph.draw(pen)
        contours = pen.contours

        if not contours:
            char_table.append((-1, 0))
            continue

        # Bounding Box for scaling
        all_pts_for_glyph = [p for c in contours for p in c]
        if not all_pts_for_glyph:
            char_table.append((-1, 0))
            continue
            
        xs = [p[0] for p in all_pts_for_glyph]
        ys = [p[1] for p in all_pts_for_glyph]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        height = max_y - min_y
        
        s = scale if height == 0 else (target_height * 1.0) / height

        first_poly_index_for_char = len(all_polys)
        poly_count_for_char = 0

        for contour in contours:
            # Transformation: scaling and normalization
            transformed_points = [
                ((p[0] - min_x) * s, (p[1] - min_y) * s) for p in contour
            ]

            # Resampling to a fixed number of points
            resampled_points = resample_polygon(
                transformed_points, target_points_per_contour
            )

            # Signed area to determine winding order
            area = signed_area(resampled_points)
            # A positive area is CCW, which is typically a "hole".
            # A negative area is CW, which is the main shape.
            is_hole = area > 0

            # Fix the winding order if necessary
            if (is_hole and area < 0) or (not is_hole and area > 0):
                resampled_points = list(reversed(resampled_points))

            # Remove points that are too close to each other
            cleaned_points = []
            if resampled_points:
                cleaned_points.append(resampled_points[0])
                for p in resampled_points[1:]:
                    last_p = cleaned_points[-1]
                    if (abs(last_p[0] - p[0]) > 1e-6 or abs(last_p[1] - p[1]) > 1e-6):
                        cleaned_points.append(p)
            
            if len(cleaned_points) < 3:
                continue

            start_idx = len(all_points)
            all_points.extend(cleaned_points)
            all_polys.append((start_idx, len(cleaned_points), is_hole))
            poly_count_for_char += 1

        char_table.append((first_poly_index_for_char, poly_count_for_char))

    return all_points, all_polys, char_table, chars


# --- GLSL Generator ---

def escape_f(f):
    """Formats a float as a string with 6 decimal places."""
    return f"{f:.6f}"


def generate_char_ids(chars):
    """Generates GLSL const int CHAR_X = index; definitions."""
    lines = []
    for i, ch in enumerate(chars):
        char_name = ch.upper() if ch.isalnum() else "SPACE"
        lines.append(f"const int CHAR_{char_name} = {i};")
    return lines


def emit_glsl(
    all_points,
    all_polys,
    char_table,
    chars,
    poly_struct_name="PolyInfo",
    char_struct_name="CharInfo",
):
    """Generates the complete GLSL code as a string."""
    out = []
    out.append("// Generated by font_to_shadertoy.py")
    out.append(f"// Characters: {chars}")
    out.append("")

    out.append(f"struct {poly_struct_name}" + " {")
    out.append("    int start;")
    out.append("    int count;")
    out.append("    bool isHole;")
    out.append("};")
    out.append("")

    out.append(f"struct {char_struct_name}" + " {")
    out.append("    int start;")
    out.append("    int count;")
    out.append("};")
    out.append("")

    for line in generate_char_ids(chars):
        out.append(line)
    out.append("")

    # Create allPoints
    out.append("const vec2 allPoints[] = vec2[](")
    out.append(
        "    " + ", ".join(f"vec2({escape_f(x)}, {escape_f(y)})" for (x, y) in all_points)
    )
    out.append(");")
    out.append("")

    # Create allPolys
    out.append("// PolyInfo(start, count, isHole)")
    out.append(
        f"const {poly_struct_name} allPolys[{len(all_polys)}] = {poly_struct_name}[]("
    )
    for start, count, is_hole in all_polys:
        is_hole_str = "true" if is_hole else "false"
        out.append(f"    {poly_struct_name}({start}, {count}, {is_hole_str}),")
    out[-1] = out[-1][:-1]
    out.append(");")
    out.append("")

    # Create charTable
    out.append("// CharInfo(startPolyIndex, polyCount) - aligned with character order")
    out.append(
        f"const {char_struct_name} charTable[{len(char_table)}] = {char_struct_name}[]("
    )
    for start, count in char_table:
        out.append(f"    {char_struct_name}({start}, {count}),")
    out[-1] = out[-1][:-1]
    out.append(");")
    out.append("")
    
    return "\n".join(out)


# --- CLI Entrypoint ---

def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Exports font glyphs to Shadertoy GLSL polygon arrays."
    )
    parser.add_argument("font", help="Path to the font file (ttf/otf)")
    parser.add_argument(
        "--chars",
        default="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        help="Characters to export (default: A-Z)",
    )
    parser.add_argument("--out", default=None, help="Output file (default: stdout)")
    parser.add_argument(
        "--target-height",
        type=float,
        default=1.0,
        help="Target height in shader units (default: 1.0)",
    )
    parser.add_argument(
        "--target-points-per-contour",
        type=int,
        default=12,
        help="Target points per glyph contour (default: 12)",
    )
    args = parser.parse_args()

    print("Loading font:", args.font)
    all_points, all_polys, char_table, chars = extract_glyph_contours(
        args.font,
        chars=args.chars,
        target_height=args.target_height,
        target_points_per_contour=args.target_points_per_contour,
    )

    glsl = emit_glsl(all_points, all_polys, char_table, chars)

    if args.out:
        with open(args.out, "w", encoding="utf8") as f:
            f.write(glsl)
        print("GLSL was written to", args.out, ".")
    else:
        print(glsl)


if __name__ == "__main__":
    main()