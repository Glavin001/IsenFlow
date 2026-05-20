import { describe, it, expect } from 'vitest';
import { FractureScheduler } from '../../src/destruction/FractureScheduler.js';

describe('FractureScheduler', () => {
  it('drains at most perFrameBudget per frame', () => {
    const s = new FractureScheduler<string>(2);
    for (const c of ['a', 'b', 'c', 'd']) s.request(c, 1);
    expect(s.drainFrame().length).toBe(2);
    expect(s.pending).toBe(2);
    expect(s.drainFrame().length).toBe(2);
    expect(s.pending).toBe(0);
  });

  it('orders by priority descending', () => {
    const s = new FractureScheduler<string>(2);
    s.request('low', 1);
    s.request('high', 10);
    s.request('mid', 5);
    expect(s.drainFrame()).toEqual(['high', 'mid']);
  });

  it('clear empties the queue', () => {
    const s = new FractureScheduler<string>(5);
    s.request('a', 1);
    s.clear();
    expect(s.pending).toBe(0);
  });
});
