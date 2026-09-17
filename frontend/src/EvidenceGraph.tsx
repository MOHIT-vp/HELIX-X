'use client'

import { Background, Controls, MiniMap, Position, ReactFlow, type Edge, type Node, type OnEdgesChange, type OnNodesChange, useEdgesState, useNodesState } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { Filter, Focus, Route, RotateCcw } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState, type CSSProperties } from 'react'
import type { EvidenceEdge, EvidenceGraph as EvidenceGraphData, EvidenceNode } from './api'

const nodeColors: Record<string, string> = { Compound: '#FFBF00', Gene: '#FFF78D', Pathway: '#467235', Disease: '#E87F24' }
const nodeTypes = ['Compound', 'Gene', 'Pathway', 'Disease']

function layout(graph: EvidenceGraphData): Node[] {
  const buckets = new Map<number, EvidenceNode[]>()
  graph.nodes.forEach((node, index) => {
    const column = node.path_positions[0] ?? index % 5
    buckets.set(column, [...(buckets.get(column) || []), node])
  })
  return graph.nodes.map((node, index) => {
    const column = node.path_positions[0] ?? index % 5
    const lane = buckets.get(column) || []
    const row = lane.findIndex(item => item.id === node.id)
    return {
      id: node.id,
      position: { x: column * 220, y: row * 112 + (column % 2 ? 38 : 0) },
      data: { label: <div className="evidence-node" style={{ '--node-color': nodeColors[node.type] || '#FFF78D' } as CSSProperties}><span>{node.type}</span><strong>{node.name}</strong><small>#{node.index}</small></div> },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
      className: `evidence-flow-node ${node.type.toLowerCase()}`,
    }
  })
}

function graphEdges(edges: EvidenceEdge[]): Edge[] {
  return edges.map(edge => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    label: edge.relationship,
    type: 'smoothstep',
    animated: edge.path_index === 0,
    className: 'evidence-flow-edge',
    labelStyle: { fill: '#FFF78D', fontSize: 10, letterSpacing: '.06em' },
    labelBgStyle: { fill: '#283F24', fillOpacity: .9 },
    labelBgPadding: [4, 3] as [number, number],
  }))
}

type Props = {
  graph: EvidenceGraphData
  selectedNode: string | null
  onSelectNode: (id: string | null) => void
}

export default function EvidenceGraphExplorer({ graph, selectedNode, onSelectNode }: Props) {
  const baseNodes = useMemo(() => layout(graph), [graph])
  const baseEdges = useMemo(() => graphEdges(graph.edges), [graph.edges])
  const [nodes, setNodes, onNodesChange] = useNodesState(baseNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(baseEdges)
  const [visibleTypes, setVisibleTypes] = useState(nodeTypes)
  const [activePath, setActivePath] = useState<string | null>(null)
  const [flow, setFlow] = useState<import('@xyflow/react').ReactFlowInstance | null>(null)
  const [selectedEdge, setSelectedEdge] = useState<EvidenceEdge | null>(null)

  useEffect(() => { setNodes(baseNodes); setEdges(baseEdges); setVisibleTypes(nodeTypes); setActivePath(null); setSelectedEdge(null) }, [baseNodes, baseEdges, setEdges, setNodes])

  useEffect(() => {
    const allowed = new Set(visibleTypes)
    const shownNodes = baseNodes.map(node => ({ ...node, hidden: !allowed.has(graph.nodes.find(item => item.id === node.id)?.type || '') }))
    const shown = new Set(shownNodes.filter(node => !node.hidden).map(node => node.id))
    const highlighted = activePath ? new Set(graph.paths.find(path => path.id === activePath)?.node_ids || []) : null
    setNodes(shownNodes.map(node => ({ ...node, className: `${node.className || ''} ${selectedNode === node.id ? 'is-selected' : ''} ${highlighted?.has(node.id) ? 'is-path' : ''}` })))
    setEdges(baseEdges.map(edge => ({ ...edge, hidden: !shown.has(edge.source) || !shown.has(edge.target), animated: Boolean(activePath && graph.paths.find(path => path.id === activePath)?.node_ids.includes(edge.source) && graph.paths.find(path => path.id === activePath)?.node_ids.includes(edge.target)), className: `${edge.className || ''} ${selectedEdge?.id === edge.id ? 'is-selected' : ''}` })))
  }, [activePath, baseEdges, baseNodes, graph.nodes, graph.paths, selectedEdge, selectedNode, setEdges, setNodes, visibleTypes])

  const toggleType = (type: string) => setVisibleTypes(current => current.includes(type) ? current.filter(item => item !== type) : [...current, type])
  const reset = () => { setVisibleTypes(nodeTypes); setActivePath(null); setSelectedEdge(null); onSelectNode(null); requestAnimationFrame(() => flow?.fitView({ padding: .22, duration: 450 })) }
  const onNodeClick = useCallback((_event: React.MouseEvent, node: Node) => { setSelectedEdge(null); onSelectNode(node.id) }, [onSelectNode])
  const onEdgeClick = useCallback((_event: React.MouseEvent, edge: Edge) => { setSelectedEdge(graph.edges.find(item => item.id === edge.id) || null); onSelectNode(null) }, [graph.edges, onSelectNode])

  return <div className="evidence-explorer">
    <div className="graph-toolbar evidence-toolbar"><span>ACTUAL API RESPONSE · BOUNDED EVIDENCE SUBGRAPH</span><div><button onClick={() => flow?.fitView({ padding: .22, duration: 450 })}><Focus size={14} /> Fit</button><button onClick={reset}><RotateCcw size={14} /> Reset</button></div></div>
    <div className="evidence-controls"><span><Filter size={13} /> Filter</span>{nodeTypes.map(type => <button key={type} className={visibleTypes.includes(type) ? 'active' : ''} onClick={() => toggleType(type)}><i style={{ background: nodeColors[type] }} />{type}</button>)}{graph.paths.length > 0 && <label><Route size={13} /><select value={activePath || ''} onChange={event => setActivePath(event.target.value || null)}><option value="">Highlight a path</option>{graph.paths.map((path, index) => <option key={path.id} value={path.id}>Path {String(index + 1).padStart(2, '0')} · {path.hops} hops</option>)}</select></label>}</div>
    <div className="evidence-flow" aria-label="Interactive graph rendered from inference API evidence">
      <ReactFlow nodes={nodes} edges={edges} onNodesChange={onNodesChange as OnNodesChange} onEdgesChange={onEdgesChange as OnEdgesChange} onNodeClick={onNodeClick} onEdgeClick={onEdgeClick} onInit={setFlow} minZoom={.25} maxZoom={2.2} fitView fitViewOptions={{ padding: .22 }} nodesDraggable panOnDrag>
        <Background color="rgba(255,247,141,.16)" gap={22} size={1} />
        <MiniMap nodeColor={node => nodeColors[graph.nodes.find(item => item.id === node.id)?.type || 'Gene']} maskColor="rgba(40,63,36,.78)" pannable zoomable />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
    {selectedEdge && <div className="edge-inspector"><span>Relationship</span><strong>{selectedEdge.relationship}</strong><small>Path {selectedEdge.path_index + 1} · click a node to inspect entity details</small></div>}
  </div>
}
