import type { WallChunk } from '../destruction/WallChunk.js';

/**
 * Registry of all chunks/bodies that participate in two-way coupling.
 * Chunk IDs are dense small ints; 0 is reserved for "no chunk".
 */
export class BodyTracker {
  private readonly byId = new Map<number, WallChunk>();
  private nextId = 1;

  register(chunkFactory: (id: number) => WallChunk): WallChunk {
    const id = this.nextId++;
    const c = chunkFactory(id);
    this.byId.set(id, c);
    return c;
  }

  get(id: number): WallChunk | undefined {
    return this.byId.get(id);
  }

  remove(id: number): void {
    this.byId.delete(id);
  }

  all(): IterableIterator<WallChunk> {
    return this.byId.values();
  }

  get size(): number {
    return this.byId.size;
  }
}
