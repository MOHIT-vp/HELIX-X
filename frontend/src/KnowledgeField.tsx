'use client'

import { Canvas, useFrame } from '@react-three/fiber'
import { Float, Line, PerspectiveCamera, Sparkles } from '@react-three/drei'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import type { EvidenceGraph } from './api'

const palette = { Compound: '#FFBF00', Disease: '#E87F24', Gene: '#FFF78D', Pathway: '#467235' }
type Point = { id: string; type: keyof typeof palette; position: [number, number, number] }
type Link = { start: Point['position']; end: Point['position']; emphasis?: boolean }

function makeConceptualField(count: number): Point[] {
  const types: Point['type'][] = ['Compound', 'Gene', 'Pathway', 'Gene', 'Disease', 'Gene', 'Pathway']
  return Array.from({ length: count }, (_, index) => {
    const arm = index % 5
    const ring = Math.floor(index / 5)
    const angle = arm * (Math.PI * 2 / 5) + ring * .43
    const radius = .75 + ring * .23
    return { id: `field-${index}`, type: types[index % types.length], position: [Math.cos(angle) * radius, Math.sin(angle * 1.5) * (1.15 + ring * .1), Math.sin(angle) * radius * .72 + Math.cos(ring * 1.13) * .35] }
  })
}

function Field({ graph, compact, selectedId, energized }: { graph?: EvidenceGraph | null; compact: boolean; selectedId?: string | null; energized: boolean }) {
  const group = useRef<THREE.Group>(null)
  const points = useMemo<Point[]>(() => {
    if (graph?.nodes.length) return graph.nodes.slice(0, 32).map((node, index) => {
      const column = node.path_positions[0] ?? index % 5
      const inColumn = graph.nodes.slice(0, index).filter(item => (item.path_positions[0] ?? 0) === column).length
      return { id: node.id, type: (node.type in palette ? node.type : 'Gene') as Point['type'], position: [(column - 2) * 1.5, 1.15 - inColumn * .8, Math.sin(index * 1.91) * .55] }
    })
    return makeConceptualField(compact ? 36 : 64)
  }, [graph, compact])
  const links = useMemo<Link[]>(() => {
    if (graph?.edges.length) return graph.edges.map(edge => {
      const start = points.find(point => point.id === edge.source)?.position
      const end = points.find(point => point.id === edge.target)?.position
      return start && end ? { start, end, emphasis: selectedId === edge.source || selectedId === edge.target } : null
    }).filter(Boolean) as Link[]
    return points.flatMap((point, index) => {
      const next = points[(index + 1) % points.length]
      const cross = index % 3 === 0 ? points[(index + 7) % points.length] : null
      return [{ start: point.position, end: next.position, emphasis: index % 8 === 0 }, ...(cross ? [{ start: point.position, end: cross.position }] : [])]
    })
  }, [graph, points, selectedId])

  useFrame((state, delta) => {
    if (!group.current) return
    group.current.rotation.y += delta * (energized ? .22 : .035)
    group.current.rotation.x = THREE.MathUtils.lerp(group.current.rotation.x, state.pointer.y * .12, .025)
    group.current.rotation.z = THREE.MathUtils.lerp(group.current.rotation.z, state.pointer.x * .08, energized ? .08 : .025)
  })

  return <group ref={group}>
    {links.map((link, index) => <Line key={`link-${index}`} points={[link.start, link.end]} color={link.emphasis ? '#FFBF00' : '#8BA66F'} transparent opacity={link.emphasis ? .72 : .19} lineWidth={link.emphasis ? 1.5 : .65} />)}
    {links.filter((_, index) => index % (energized ? 3 : 7) === 0).map((link, index) => <Signal key={`signal-${index}`} start={link.start} end={link.end} offset={index / 7} energized={energized} />)}
    {points.map(point => <Float key={point.id} speed={.45} rotationIntensity={0} floatIntensity={.12} floatingRange={[-.06, .06]}><mesh position={point.position}><sphereGeometry args={[point.type === 'Disease' ? .13 : point.type === 'Compound' ? .105 : .075, 14, 14]} /><meshBasicMaterial color={selectedId === point.id ? '#E87F24' : palette[point.type]} transparent opacity={selectedId && selectedId !== point.id ? .28 : .95} /></mesh></Float>)}
  </group>
}

function Signal({ start, end, offset, energized }: { start: [number, number, number]; end: [number, number, number]; offset: number; energized: boolean }) {
  const signal = useRef<THREE.Mesh>(null)
  useFrame(({ clock }) => { const t = (clock.getElapsedTime() * (energized ? .62 : .17) + offset) % 1; signal.current?.position.lerpVectors(new THREE.Vector3(...start), new THREE.Vector3(...end), t) })
  return <mesh ref={signal}><sphereGeometry args={[.025, 8, 8]} /><meshBasicMaterial color="#FFBF00" /></mesh>
}

function CameraDrift({ compact, energized }: { compact: boolean; energized: boolean }) {
  useFrame(({ camera, clock }) => {
    const scrollFactor = typeof window === 'undefined' ? 0 : Math.min(window.scrollY / Math.max(window.innerHeight * 5, 1), 1)
    const base = compact ? 8.2 : 9.5
    const targetZ = base - scrollFactor * (compact ? .9 : 1.7) - (energized ? 1.2 : 0)
    camera.position.z = THREE.MathUtils.lerp(camera.position.z, targetZ, energized ? .07 : .018)
    camera.position.x = THREE.MathUtils.lerp(camera.position.x, Math.sin(clock.getElapsedTime() * .1) * .3, .02)
    camera.lookAt(0, 0, 0)
  })
  return null
}

export default function KnowledgeField({ graph, compact = false, selectedId, energized = false }: { graph?: EvidenceGraph | null; compact?: boolean; selectedId?: string | null; energized?: boolean }) {
  return <div className={`knowledge-field ${compact ? 'compact' : ''}`}><Canvas dpr={[1, 1.5]} gl={{ antialias: true, alpha: true }}><PerspectiveCamera makeDefault position={[0, 0, compact ? 8.2 : 9.5]} fov={42} /><fog attach="fog" args={['#283F24', 5, 15]} /><ambientLight intensity={.6} /><pointLight position={[2, 3, 3]} color="#FFBF00" intensity={energized ? 5 : 3} distance={8} /><Sparkles count={compact ? 55 : 90} scale={[7, 5, 4]} size={1.2} speed={energized ? .6 : .15} color="#FFF78D" opacity={.45} /><CameraDrift compact={compact} energized={energized} /><Field graph={graph} compact={compact} selectedId={selectedId} energized={energized} /></Canvas><div className="field-key"><span><i className="dot compound" />COMPOUND</span><span><i className="dot gene" />GENE</span><span><i className="dot pathway" />PATHWAY</span><span><i className="dot disease" />DISEASE</span></div></div>
}
