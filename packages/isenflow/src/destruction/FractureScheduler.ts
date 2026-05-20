/**
 * Rate-limits fracture events per frame so a single catastrophic flood pulse
 * cannot stall the renderer by spawning hundreds of chunks at once.
 */

export interface FractureRequest<TChunk> {
  chunk: TChunk;
  priority: number; // higher = more urgent
}

export class FractureScheduler<TChunk> {
  private queue: FractureRequest<TChunk>[] = [];
  constructor(private readonly perFrameBudget: number) {}

  request(chunk: TChunk, priority: number): void {
    this.queue.push({ chunk, priority });
  }

  /** Drain up to perFrameBudget items, highest priority first. */
  drainFrame(): TChunk[] {
    if (this.queue.length === 0) return [];
    this.queue.sort((a, b) => b.priority - a.priority);
    const taken = this.queue.splice(0, this.perFrameBudget);
    return taken.map((r) => r.chunk);
  }

  get pending(): number {
    return this.queue.length;
  }

  clear(): void {
    this.queue.length = 0;
  }
}
