// SPDX-License-Identifier: Apache-2.0
// Open-model load: constant arrival rate against hybrid + rerank, shipped defaults.
// Env: BASE_URL, API_KEY, COLLECTION_ID, QUERIES_FILE (JSON array of strings), RATE (req/s),
// DURATION (k6 duration, e.g. 5m).
import http from "k6/http";
import { check } from "k6";

const queries = JSON.parse(open(__ENV.QUERIES_FILE));
const rate = Number(__ENV.RATE || 2);

export const options = {
  scenarios: {
    open_model: {
      executor: "constant-arrival-rate",
      rate: rate,
      timeUnit: "1s",
      duration: __ENV.DURATION || "5m",
      preAllocatedVUs: 10,
      maxVUs: 50,
    },
  },
  summaryTrendStats: ["avg", "min", "med", "p(90)", "p(95)", "p(99)", "max"],
};

export default function () {
  const query = queries[__ITER % queries.length];
  const res = http.post(
    `${__ENV.BASE_URL}/v1/collections/${__ENV.COLLECTION_ID}/search`,
    JSON.stringify({ query: query, mode: "hybrid", rerank: true, k: 8 }),
    {
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${__ENV.API_KEY}`,
      },
    },
  );
  check(res, { "status 200": (r) => r.status === 200 });
}
