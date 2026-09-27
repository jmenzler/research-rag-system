/**
 * Pure BFS depth and shortest-path computation for Sigma graph walk (D-02, D-03).
 *
 * No React, no sigma runtime imports — fully unit-testable in jsdom.
 */

export interface GraphNodeInput {
  corpusId: number;
  isSeed: boolean;
}

export interface GraphEdgeInput {
  fromId: number;
  toId: number;
}

/**
 * Compute BFS depths from the seed node (the node with isSeed:true).
 *
 * Edges are treated as undirected: a→b makes both b reachable from a
 * and a reachable from b.
 *
 * Returns a Map<corpusId, depth> where depth 0 = seed.
 * Disconnected nodes are NOT present in the returned Map.
 * If no node has isSeed:true, returns an empty Map.
 */
export function computeBFSDepths(
  nodes: GraphNodeInput[],
  edges: GraphEdgeInput[],
): Map<number, number> {
  const seedId = nodes.find((n) => n.isSeed)?.corpusId;
  if (seedId === undefined) return new Map();

  const adj = new Map<number, number[]>();
  for (const n of nodes) adj.set(n.corpusId, []);
  for (const e of edges) {
    adj.get(e.fromId)?.push(e.toId);
    adj.get(e.toId)?.push(e.fromId);
  }

  const depths = new Map<number, number>();
  depths.set(seedId, 0);
  const queue: number[] = [seedId];
  while (queue.length > 0) {
    const curr = queue.shift()!;
    const d = depths.get(curr)!;
    for (const neighbor of adj.get(curr) ?? []) {
      if (!depths.has(neighbor)) {
        depths.set(neighbor, d + 1);
        queue.push(neighbor);
      }
    }
  }
  return depths;
}

/**
 * Compute shortest path from seedId to targetId over an undirected adjacency.
 *
 * Returns [seedId, ..., targetId] for a reachable target.
 * Returns [seedId] when seedId === targetId.
 * Returns [] when targetId is unreachable.
 */
export function computeShortestPath(
  seedId: number,
  targetId: number,
  edges: GraphEdgeInput[],
): number[] {
  if (seedId === targetId) return [seedId];

  const adj = new Map<number, number[]>();

  const ensureNode = (id: number) => {
    if (!adj.has(id)) adj.set(id, []);
  };

  for (const e of edges) {
    ensureNode(e.fromId);
    ensureNode(e.toId);
    adj.get(e.fromId)!.push(e.toId);
    adj.get(e.toId)!.push(e.fromId);
  }

  const parent = new Map<number, number>();
  const visited = new Set<number>();
  visited.add(seedId);
  const queue: number[] = [seedId];

  while (queue.length > 0) {
    const curr = queue.shift()!;
    for (const neighbor of adj.get(curr) ?? []) {
      if (!visited.has(neighbor)) {
        visited.add(neighbor);
        parent.set(neighbor, curr);
        if (neighbor === targetId) {
          const path: number[] = [];
          let node: number | undefined = targetId;
          while (node !== undefined) {
            path.unshift(node);
            node = parent.get(node);
          }
          return path;
        }
        queue.push(neighbor);
      }
    }
  }

  return [];
}
