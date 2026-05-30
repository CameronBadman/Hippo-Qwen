const state = {
  lastSearch: null,
  graph: { nodes: {}, edges: {} },
}

const $ = (id) => document.getElementById(id)
const API_PREFIX = "/api"

async function api(path, options = {}) {
  const response = await fetch(`${API_PREFIX}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  })
  const contentType = response.headers.get("content-type") || ""
  const raw = await response.text()
  if (!contentType.includes("application/json")) {
    throw new Error(`Expected JSON from ${path}, got ${contentType || "unknown"}: ${raw.slice(0, 80)}`)
  }
  const body = JSON.parse(raw)
  if (!response.ok) throw new Error(body.error || response.statusText)
  return body
}

function resultList(target, results) {
  target.innerHTML = ""
  for (const result of results || []) {
    const li = document.createElement("li")
    li.textContent = result.node.text
    const score = document.createElement("span")
    score.className = "score"
    score.textContent = `score ${result.score.toFixed(3)} | vector ${result.vector_score.toFixed(3)} | graph ${result.graph_score.toFixed(3)}`
    li.appendChild(score)
    target.appendChild(li)
  }
}

async function refreshGraph(activeIds = []) {
  state.graph = await api("/graph")
  drawGraph(new Set(activeIds))
}

function drawGraph(activeIds) {
  const svg = $("graphSvg")
  svg.replaceChildren()
  const nodes = Object.values(state.graph.nodes || {})
  const edges = Object.values(state.graph.edges || {})
  $("graphStats").textContent = `${nodes.length} nodes | ${edges.length} edges`
  if (nodes.length === 0) return

  const width = 1100
  const height = 460
  const cx = width / 2
  const cy = height / 2
  const radius = Math.min(width, height) * 0.38
  const positions = new Map()
  nodes.forEach((node, index) => {
    const angle = (index / nodes.length) * Math.PI * 2 - Math.PI / 2
    positions.set(node.id, {
      x: cx + Math.cos(angle) * radius,
      y: cy + Math.sin(angle) * radius,
    })
  })

  for (const edge of edges) {
    const a = positions.get(edge.source)
    const b = positions.get(edge.target)
    if (!a || !b) continue
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line")
    line.setAttribute("x1", a.x)
    line.setAttribute("y1", a.y)
    line.setAttribute("x2", b.x)
    line.setAttribute("y2", b.y)
    line.setAttribute("class", "edge")
    line.setAttribute("opacity", String(Math.max(0.15, Math.min(0.9, edge.weight))))
    svg.appendChild(line)
  }

  for (const node of nodes) {
    const point = positions.get(node.id)
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle")
    circle.setAttribute("cx", point.x)
    circle.setAttribute("cy", point.y)
    circle.setAttribute("r", 16 + node.importance * 8)
    circle.setAttribute("class", activeIds.has(node.id) ? "node active" : "node")
    svg.appendChild(circle)

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text")
    label.setAttribute("x", point.x + 22)
    label.setAttribute("y", point.y + 4)
    label.setAttribute("class", "label")
    label.textContent = shortText(node.text)
    svg.appendChild(label)
  }
}

function shortText(text) {
  return text.length > 34 ? `${text.slice(0, 31)}...` : text
}

$("memoryForm").addEventListener("submit", async (event) => {
  event.preventDefault()
  const project = $("memoryProject").value.trim()
  const payload = {
    text: $("memoryText").value,
    importance: Number($("memoryImportance").value || "0.5"),
    metadata: project ? { project } : {},
  }
  try {
    const response = await api("/memories", { method: "POST", body: JSON.stringify(payload) })
    $("memoryText").value = ""
    await refreshGraph([response.node.id, ...response.edges.flatMap((edge) => [edge.source, edge.target])])
  } catch (error) {
    alert(error.message)
  }
})

$("searchForm").addEventListener("submit", async (event) => {
  event.preventDefault()
  try {
    const response = await api("/search", {
      method: "POST",
      body: JSON.stringify({
        query: $("queryText").value,
        limit: Number($("limit").value || "8"),
        budget: Number($("budget").value || "1800"),
      }),
    })
    state.lastSearch = response
    resultList($("vectorResults"), response.vector_results)
    resultList($("graphResults"), response.graph_results)
    await refreshGraph(response.context_node_ids || [])
  } catch (error) {
    alert(error.message)
  }
})

async function sendFeedback(outcome) {
  if (!state.lastSearch) return
  await api("/feedback", {
    method: "POST",
    body: JSON.stringify({
      outcome,
      node_ids: state.lastSearch.context_node_ids || [],
      edge_ids: state.lastSearch.context_edge_ids || [],
    }),
  })
  await refreshGraph(state.lastSearch.context_node_ids || [])
}

$("helpfulBtn").addEventListener("click", () => sendFeedback("helpful"))
$("ignoredBtn").addEventListener("click", () => sendFeedback("ignored"))
$("correctedBtn").addEventListener("click", () => sendFeedback("corrected"))
$("decayBtn").addEventListener("click", async () => {
  await api("/maintenance/decay", {
    method: "POST",
    body: JSON.stringify({ factor: 0.98, gc_threshold: 0.05 }),
  })
  await refreshGraph()
})

refreshGraph().catch((error) => console.error(error))
