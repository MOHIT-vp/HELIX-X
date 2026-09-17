import type { Metadata } from 'next'
import '../src/styles.css'
import '../src/immersive.css'

export const metadata: Metadata = {
  title: 'HELIXX | Biomedical Intelligence',
  description: 'Graph neural reasoning for compound-disease hypothesis generation.',
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>
}
