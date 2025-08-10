# Font to Shadertoy GLSL Exporter

This is a Python script designed to convert TrueType (`.ttf`) or OpenType (`.otf`) font files into GLSL arrays that can be used for rendering vector fonts directly in a Shadertoy fragment shader.

The script simplifies the process of getting font data into a format that is ready to use on platforms like Shadertoy, where you often need to represent geometric data as arrays.

### Output Structure

The script generates a GLSL file containing three main data structures:

* `vec2 allPoints[]`: A flat array of all the `(x, y)` points for every glyph contour.
* `const PolyInfo allPolys[]`: An array of structs that define each polygon (contour) by referencing a start index and a count from `allPoints`. It also includes a `bool` to indicate if the polygon is a hole.
* `const CharInfo charTable[]`: An array of structs that map each exported character to its corresponding polygons in the `allPolys` array.

### Installation

To run this script, you need to install the `fonttools` library.

```bash
pip install fonttools
````

### Usage

Run the script from your command line. The basic usage requires you to provide a path to a font file.

```bash
python font_to_shadertoy.py <path/to/your/font.ttf>
```

#### Command-Line Arguments

  * `<font>` (required): The path to the input font file (`.ttf` or `.otf`).
  * `--out <filename>` (optional): The path to the output GLSL file. If not provided, the GLSL code will be printed to the console.
  * `--chars <string>` (optional): A string containing all the characters you want to export. By default, it exports "ABCDEFGHIJKLMNOPQRSTUVWXYZ".
  * `--target-height <float>` (optional): The desired height of the glyphs in shader units. The default is `1.0`.
  * `--target-points-per-contour <int>` (optional): The number of points to resample each contour to. The default is `12`.

#### Example

To export the uppercase letters from `Roboto-Regular.ttf` and save the output to `glyphs.glsl`, you would run:

```bash
python font_to_shadertoy.py Roboto-Regular.ttf --out glyphs.glsl
```