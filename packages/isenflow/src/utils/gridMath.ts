/**
 * Grid <-> world coordinate conversions.
 *
 * Grid: integer (i, j) where i indexes world-X and j indexes world-Z.
 * Cell (i, j) covers world rect [origin + (i*dx, j*dx), origin + ((i+1)*dx, (j+1)*dx)].
 * Cell center samples at origin + ((i+0.5)*dx, (j+0.5)*dx).
 */

export interface GridDescriptor {
  /** Number of cells along X. */
  readonly width: number;
  /** Number of cells along Z. */
  readonly height: number;
  /** Cell spacing in meters. */
  readonly dx: number;
  /** World-space (X, Z) of cell (0, 0)'s minimum corner. */
  readonly origin: readonly [number, number];
}

export interface Aabb2 {
  readonly minX: number;
  readonly minZ: number;
  readonly maxX: number;
  readonly maxZ: number;
}

export interface CellRange {
  readonly i0: number;
  readonly j0: number;
  readonly i1: number;
  readonly j1: number;
}

export function worldToGrid(grid: GridDescriptor, x: number, z: number): readonly [number, number] {
  return [
    Math.floor((x - grid.origin[0]) / grid.dx),
    Math.floor((z - grid.origin[1]) / grid.dx),
  ];
}

export function gridToWorldCenter(
  grid: GridDescriptor,
  i: number,
  j: number,
): readonly [number, number] {
  return [grid.origin[0] + (i + 0.5) * grid.dx, grid.origin[1] + (j + 0.5) * grid.dx];
}

export function clipAabbToGrid(grid: GridDescriptor, aabb: Aabb2): CellRange {
  const [i0Raw, j0Raw] = worldToGrid(grid, aabb.minX, aabb.minZ);
  const [i1Raw, j1Raw] = worldToGrid(grid, aabb.maxX, aabb.maxZ);
  return {
    i0: clamp(i0Raw, 0, grid.width - 1),
    j0: clamp(j0Raw, 0, grid.height - 1),
    i1: clamp(i1Raw, 0, grid.width - 1),
    j1: clamp(j1Raw, 0, grid.height - 1),
  };
}

export function indexOf(grid: GridDescriptor, i: number, j: number): number {
  return j * grid.width + i;
}

export function isInBounds(grid: GridDescriptor, i: number, j: number): boolean {
  return i >= 0 && j >= 0 && i < grid.width && j < grid.height;
}

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Cell-area in m² — used by force/buoyancy integrals. */
export function cellArea(grid: GridDescriptor): number {
  return grid.dx * grid.dx;
}
