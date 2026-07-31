# Local CT evidence crops - PHercParis4 exported ranks 7 and 11

This artifact holds two small level-0 CT neighbourhoods extracted at the positions that two
exported render candidates map to. It exists so those candidates can be inspected against
real volumetric data instead of only against the 2D render

Two candidates are covered:

| Exported rank | Component ID |
|---:|---:|
| 7 | 566 |
| 11 | 690 |

Exported rank and component ID are different identifiers. The rank is the position in the
exported candidate ordering; the component ID labels the connected component in the
render-analysis output. Both are given so a crop can be matched back through either

## What these are

Each crop is a 64<sup>3</sup> level-0 neighbourhood centred on the position that the exported
candidate maps to, after the tifxyz surface coordinate is carried into the CT volume frame by
the inverse registration

That coordinate interpretation is **supported, not verified**. It rests on two earlier
results: the registration matrix direction established by landmark residual testing
(mean 0.92 voxels one way, ~27,459 the other), and a chunk-presence test in which the native
reading landed in chunks the volume does not store while the transformed reading landed in
dense material. Chunk content cannot prove provenance, so the frame remains an
interpretation

These crops are therefore **evidence for inspection, not adjudication**. Nothing here
establishes that either candidate is a confirmed sheet skip, a reconstruction error, a false
positive, or volumetrically validated

## Crop integrity

| Check | rank 7 | rank 11 |
|---|---|---|
| mapped array index `(z, y, x)` | `(64777, 20078, 7844)` | `(63151, 20406, 14780)` |
| bounds `z` | `[64745, 64809)` | `[63119, 63183)` |
| bounds `y` | `[20046, 20110)` | `[20374, 20438)` |
| bounds `x` | `[7812, 7876)` | `[14748, 14812)` |
| clipped against volume | no | no |
| crop shape | `(64, 64, 64)` | `(64, 64, 64)` |
| dtype | uint8 | uint8 |
| mapped point local index | `(32, 32, 32)` | `(32, 32, 32)` |
| every voxel covered by a fetched chunk | yes | yes |

Bounds are half-open. Neither crop required clipping; both sit well inside the volume extent

## Measurements

### rank 7, component ID 566

| Metric | Value |
|---|---|
| min / max | 0 / 201 |
| mean / std | 67.363373 / 38.150181 |
| nonzero fraction | 0.99998093 (262,139 of 262,144) |
| centre voxel value | **29** |
| 9×9×9 shape | `[9, 9, 9]`, unclipped, 729 voxels |
| 9×9×9 min / max | 4 / 98 |
| 9×9×9 mean / std | 43.440329 / 18.909127 |
| 9×9×9 nonzero fraction | 1.0 |

### rank 11, component ID 690

| Metric | Value |
|---|---|
| min / max | 0 / 235 |
| mean / std | 69.894554 / 37.114030 |
| nonzero fraction | 0.99997711 (262,138 of 262,144) |
| centre voxel value | **105** |
| 9×9×9 shape | `[9, 9, 9]`, unclipped, 729 voxels |
| 9×9×9 min / max | 72 / 151 |
| 9×9×9 mean / std | 114.116598 / 16.150573 |
| 9×9×9 nonzero fraction | 1.0 |

## Visual observations

Stated as observations about image content only. No classification is drawn from them

Both mapped points fall within populated regions of the masked reconstruction, where laminar
structure is visible. Nonzero fraction is essentially 1.0 in each crop, and alternating
bright sheet-like bands and darker inter-sheet gaps are visible across all three orthogonal
planes and persist across the +/-12-voxel stacks

**Rank 7.** The mapped point sits in a locally dark region immediately adjacent to a bright
lamina that runs steeply through the crop. Its 9×9×9 mean of 43.44 is well below the crop
mean of 67.36, so the neighbourhood is lower-density than the surrounding block. In the XZ
stack the nearby bright band shifts laterally across the offsets while the marked position
stays in darker material

**Rank 11.** The mapped point sits on brighter material. Its 9×9×9 mean of 114.12 is well
above the crop mean of 69.89. All values in the immediate 9×9×9 neighbourhood are nonzero and
range from 72 to 151. The surrounding structure is denser and more tightly folded than in the
rank 7 crop

The difference between the two is recorded as a measured intensity contrast, not as a verdict
about either candidate. A single voxel and a 9<sup>3</sup> window cannot distinguish a
surface sitting in an inter-sheet gap from ordinary local density variation, and no surface
normal, no sheet assignment, and no comparison against the tifxyz surface itself was computed
here

## Outputs

| File | Contents |
|---|---|
| `README.md` | this file |
| `summary.json` | machine-readable record: chunks, bytes, bounds, shapes, all statistics |
| `rank-7-level0-crop.npy` | 64<sup>3</sup> uint8 crop, `[z, y, x]` |
| `rank-7-orthogonal.png` | centre XY, XZ, YZ slices |
| `rank-7-xy-stack.png` | XY at offsets -12, -6, 0, +6, +12 along `z` |
| `rank-7-xz-stack.png` | XZ at offsets -12, -6, 0, +6, +12 along `y` |
| `rank-7-yz-stack.png` | YZ at offsets -12, -6, 0, +6, +12 along `x` |
| `rank-11-level0-crop.npy` | 64<sup>3</sup> uint8 crop, `[z, y, x]` |
| `rank-11-orthogonal.png` | centre XY, XZ, YZ slices |
| `rank-11-xy-stack.png` | XY at offsets -12, -6, 0, +6, +12 along `z` |
| `rank-11-xz-stack.png` | XZ at offsets -12, -6, 0, +6, +12 along `y` |
| `rank-11-yz-stack.png` | YZ at offsets -12, -6, 0, +6, +12 along `x` |

The PNG views, NumPy crops, and `summary.json` form the published evidence package

Rendering: fixed grayscale range 0-255, `interpolation="nearest"`, no denoising, no contrast
normalisation, no resampling. The mapped point is marked with a red crosshair on every panel;
on off-centre stack panels the crosshair marks the in-plane position of the mapped point, with
the offset along the plane normal given in the title. Axis labels follow the plane: XY has
columns `x` and rows `y`; XZ has columns `x` and rows `z`; YZ has columns `y` and rows `z`.
Every panel is labelled with exported rank, component ID, plane, offset and pyramid level

## Unresolved limitations

1. The coordinate frame is supported, not verified. If the transformed reading is wrong, both
   crops are the wrong neighbourhoods
2. The render-to-tifxyz identity orientation assumption from the original mapping is still
   unproven and still upstream of everything here
3. The normalized tifxyz bounding box is not fully explained by the fixed volume's bounds: it
   reaches `z = 17014.27`, above the fixed volume's `z` extent of 14376. The two candidate
   points fit, but the full surface bbox does not. Unexplained
4. Two candidates, one 64<sup>3</sup> neighbourhood each, out of 200 exported. Not a sample
   that characterises the detector or the surface
5. No surface normal, no through-thickness profile, no sheet-membership test, and no overlay
   against the tifxyz surface was computed. The crops show what is in the volume at those
   positions, nothing about whether the surface label there is correct
6. Intensities are the exported uint8 window of the original reconstruction, not calibrated
   density

## Not claimed

- confirmed sheet skip
- confirmed reconstruction error
- confirmed false positive
- detector validation
- volumetric validation
- verified coordinate provenance
- solved registration
- proven render-to-tifxyz orientation
