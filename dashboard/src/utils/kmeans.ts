export function kmeans(
  data: number[][],
  k: number,
  maxIter = 100
): { labels: number[]; centroids: number[][] } {
  const n = data.length;
  const dim = data[0].length;

  // Random init
  const centroids = data.sort(() => Math.random() - 0.5).slice(0, k);
  let labels = new Array(n).fill(0);

  for (let iter = 0; iter < maxIter; iter++) {
    // Assign
    let changed = false;
    for (let i = 0; i < n; i++) {
      let minDist = Infinity;
      let best = 0;
      for (let j = 0; j < k; j++) {
        const d = euclidean(data[i], centroids[j]);
        if (d < minDist) {
          minDist = d;
          best = j;
        }
      }
      if (labels[i] !== best) {
        labels[i] = best;
        changed = true;
      }
    }
    if (!changed) break;

    // Update centroids
    const sums = Array.from({ length: k }, () => new Array(dim).fill(0));
    const counts = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
      const l = labels[i];
      for (let d = 0; d < dim; d++) sums[l][d] += data[i][d];
      counts[l]++;
    }
    for (let j = 0; j < k; j++) {
      if (counts[j] > 0) {
        for (let d = 0; d < dim; d++) centroids[j][d] = sums[j][d] / counts[j];
      }
    }
  }

  return { labels, centroids };
}

function euclidean(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return Math.sqrt(sum);
}
