'use client'

import { useEffect } from 'react'
import Lenis from 'lenis'

export default function SmoothScroll() {
  useEffect(() => { const lenis = new Lenis({ lerp: .08, smoothWheel: true }); let frame = 0; const raf = (time: number) => { lenis.raf(time); frame = requestAnimationFrame(raf) }; frame = requestAnimationFrame(raf); return () => { cancelAnimationFrame(frame); lenis.destroy() } }, [])
  return null
}
