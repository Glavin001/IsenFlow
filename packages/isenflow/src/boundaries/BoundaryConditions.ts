/**
 * Per-cell boundary type. Sampled inside the SWE flux kernel.
 *
 * Spec §15 — five canonical boundary conditions.
 */

export const BoundaryType = {
  /** Interior cell — normal SWE update applies. */
  Interior: 0,
  /** Reflective wall — outgoing flux clamped to zero. Mass conserved. */
  Closed: 1,
  /** Free outflow — flux exits freely; mass leaves the domain. */
  Open: 2,
  /** Damped — applies artificial velocity damping to absorb waves. */
  Sponge: 3,
  /** Constant inflow — water is injected at a fixed depth each frame. */
  Inflow: 4,
  /** Sea-level clamp — h pinned to a target depth each frame. */
  Sea: 5,
} as const;

export type BoundaryTypeValue = (typeof BoundaryType)[keyof typeof BoundaryType];

export function isReflective(b: BoundaryTypeValue): boolean {
  return b === BoundaryType.Closed;
}

export function isAbsorbing(b: BoundaryTypeValue): boolean {
  return b === BoundaryType.Open || b === BoundaryType.Sponge;
}

export function isSourceLike(b: BoundaryTypeValue): boolean {
  return b === BoundaryType.Inflow || b === BoundaryType.Sea;
}
