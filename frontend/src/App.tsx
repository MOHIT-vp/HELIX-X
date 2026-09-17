import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { motion } from 'framer-motion'
import { gsap } from 'gsap'
import { Activity, ArrowRight, Beaker, ChevronRight, CircleHelp, Dna, FlaskConical, GitBranch, Layers3, Search, ShieldCheck, Sparkles, X } from 'lucide-react'
import { api, Candidate, CandidateResponse, Entity, EvidenceGraph, Methodology, Prediction, Status } from './api'
import KnowledgeField from './KnowledgeField'
import EvidenceGraphExplorer from './EvidenceGraph'
import SmoothScroll from './SmoothScroll'
import GrainOverlay from './GrainOverlay'

type Page = 'discover' | 'graph' | 'candidates' | 'methodology'

function App() {
  const [page, setPage] = useState<Page>('discover')
  const [status, setStatus] = useState<Status | null>(null)
  const [compounds, setCompounds] = useState<Entity[]>([])
  const [diseases, setDiseases] = useState<Entity[]>([])
  const [methodology, setMethodology] = useState<Methodology | null>(null)
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [candidateResponse, setCandidateResponse] = useState<CandidateResponse | null>(null)
  const [drawer, setDrawer] = useState<{ entity: Entity; source: Entity; direction: 'diseases' | 'compounds' } | null>(null)
  const [loading, setLoading] = useState(true)
  const [transitioning, setTransitioning] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    Promise.all([api.status(), api.compounds(), api.diseases(), api.methodology()])
      .then(([nextStatus, nextCompounds, nextDiseases, nextMethodology]) => { setStatus(nextStatus); setCompounds(nextCompounds); setDiseases(nextDiseases); setMethodology(nextMethodology) })
      .catch((reason: Error) => setError(reason.message === 'Failed to fetch' ? 'The local inference API is unavailable. Start FastAPI to load entities and run hypotheses.' : reason.message))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (!transitioning) return
    const context = gsap.context(() => {
      gsap.fromTo('.hero-network', { scale: .94, opacity: .62 }, { scale: 1, opacity: 1, duration: .82, ease: 'power3.out' })
    })
    return () => context.revert()
  }, [transitioning])

  const inspect = async (candidate: Candidate, response: CandidateResponse) => {
    setDrawer({ entity: candidate.entity, source: response.source, direction: response.direction })
    try {
      const compound = response.direction === 'diseases' ? response.source : candidate.entity
      const disease = response.direction === 'diseases' ? candidate.entity : response.source
      const result = await api.predict(compound.id, disease.id)
      setPrediction(result)
      setPage('graph')
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not load evidence') }
  }

  const runPrediction = async (compound: Entity, disease: Entity) => {
    setError(''); setLoading(true); setTransitioning(true)
    window.scrollTo({ top: 0, behavior: 'smooth' })
    try {
      const result = await api.predict(compound.id, disease.id)
      setPrediction(result)
      await new Promise(resolve => window.setTimeout(resolve, 850))
      setPage('graph')
    }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Prediction unavailable') }
    finally { setLoading(false); setTransitioning(false) }
  }

  return <div className="app-shell"><SmoothScroll /><GrainOverlay />
    <Header page={page} setPage={setPage} status={status} />
    {status && page !== 'discover' && <StatusStrip status={status} />}
    {error && <div className="error-banner"><ShieldCheck size={16} />{error}<button onClick={() => setError('')} aria-label="Dismiss error"><X size={16} /></button></div>}
    {loading && !status ? <LoadingState /> : <main>
      {page === 'discover' && <Discover compounds={compounds} diseases={diseases} onAnalyze={runPrediction} prediction={prediction} transitioning={transitioning} demo={status?.mode === 'demo'} unavailable={status?.status === 'unavailable'} inferenceReady={Boolean(status?.inference_ready)} />}
      {page === 'graph' && <GraphPage prediction={prediction} onBack={() => setPage('discover')} />}
      {page === 'candidates' && <Candidates compounds={compounds} diseases={diseases} response={candidateResponse} setResponse={setCandidateResponse} onInspect={inspect} inferenceReady={Boolean(status?.inference_ready)} />}
      {page === 'methodology' && <MethodologyPage methodology={methodology} status={status} />}
    </main>}
    {drawer && <div className="drawer"><button className="drawer-close" onClick={() => setDrawer(null)} aria-label="Close inspector"><X size={18} /></button><div className="eyebrow">Candidate inspector</div><div className="drawer-type">{drawer.entity.type}</div><h2>{drawer.entity.name}</h2><div className="inspector-line"><span>Source</span><strong>{drawer.source.name}</strong></div><div className="inspector-line"><span>Workflow</span><strong>{drawer.direction === 'diseases' ? 'Compound → diseases' : 'Disease → compounds'}</strong></div><button className="primary wide" onClick={() => { setDrawer(null); setPage('graph') }}>View evidence graph <ArrowRight size={16} /></button></div>}
    <Footer demo={status?.graph_source === 'synthetic_demo'} />
  </div>
}

function Header({ page, setPage, status }: { page: Page; setPage: (page: Page) => void; status: Status | null }) {
  const links: [Page, string][] = [['discover', 'Discover'], ['graph', 'Graph'], ['candidates', 'Hypotheses'], ['methodology', 'Methodology']]
  const productionReady = Boolean(status?.inference_ready && status.model_loaded && status.graph_source === 'hetionet')
  const unavailable = status?.status === 'unavailable'
  return <header className="top-nav"><button className="brand" onClick={() => setPage('discover')}><span className="brand-symbol"><GitBranch size={17} /></span><span>HELIXX<small>Biomedical intelligence</small></span></button><nav>{links.map(([key, label]) => <button key={key} className={page === key ? 'nav-link active' : 'nav-link'} onClick={() => setPage(key)}>{label}</button>)}</nav><div className="nav-status"><span className={productionReady ? 'pulse ready' : 'pulse warning'} />{productionReady ? 'SYSTEM READY' : unavailable ? 'INFERENCE UNAVAILABLE' : 'DEMO MODE'}</div></header>
}

function StatusStrip({ status }: { status: Status }) {
  return <section className="status-strip"><div><span className="status-label">GRAPH</span><strong>{status.graph_label}</strong></div><div><span className="status-label">MODEL</span><strong>{status.model_loaded ? 'Checkpoint loaded' : 'Checkpoint unavailable'}</strong></div><div><span className="status-label">EMBEDDINGS</span><strong>{status.embeddings_cached ? 'Cached' : 'Unavailable'}</strong></div><div><span className="status-label">SCOPE</span><strong>{status.node_count.toLocaleString()} nodes · {status.edge_count.toLocaleString()} edges</strong></div></section>
}

function Discover({ compounds, diseases, onAnalyze, prediction, transitioning, demo, unavailable, inferenceReady }: { compounds: Entity[]; diseases: Entity[]; onAnalyze: (compound: Entity, disease: Entity) => void; prediction: Prediction | null; transitioning: boolean; demo: boolean; unavailable: boolean; inferenceReady: boolean }) {
  const [compound, setCompound] = useState<Entity | null>(compounds[0] || null)
  const [disease, setDisease] = useState<Entity | null>(diseases[0] || null)
  useEffect(() => { if (!compound && compounds[0]) setCompound(compounds[0]); if (!disease && diseases[0]) setDisease(diseases[0]) }, [compounds, diseases])
  return <>
    <section className="hero" id="network"><div className="hero-copy"><div className="eyebrow">A computational biology laboratory</div><h1>DISCOVER<br />WHAT THE<br /><em>BIOMEDICAL</em><br />GRAPH KNOWS.</h1><p>Graph neural reasoning for discovering potential compound–disease relationships in a living biomedical knowledge field.</p><div className="hero-actions"><button className="primary" onClick={() => document.getElementById('workspace')?.scrollIntoView({ behavior: 'smooth' })}>Explore the graph <ArrowRight size={17} /></button><button className="ghost" onClick={() => document.getElementById('story')?.scrollIntoView({ behavior: 'smooth' })}>Enter the field</button></div></div><HeroNetwork energized={transitioning} /></section>
    {(demo || unavailable) && <div className="demo-alert"><Activity size={17} /><div><strong>{unavailable ? 'BIOMEDICAL INFERENCE ENGINE UNAVAILABLE' : 'DEMO MODE · SYNTHETIC GRAPH ACTIVE'}</strong><span>{unavailable ? 'The trained model checkpoint is not available. Graph metadata and methodology remain available, but model-generated hypotheses are disabled.' : 'Predictions are for interface demonstration only and are not scientifically meaningful. Add the real graph and checkpoint artefacts to enable validated inference.'}</span></div></div>}
    <section className="workspace-section" id="workspace"><div className="section-heading"><div><div className="eyebrow">04 · Hypothesis generation</div><h2>Find a connection.</h2></div><span className="mono">COMPOUND → TREATS → DISEASE</span></div><p className="workspace-lede">Ask the model to surface the evidence neighborhood behind a possible therapeutic relationship.</p><div className="selector-grid"><EntitySelector label="Compound" entity={compound} options={compounds} onChange={setCompound} icon={<Beaker size={17} />} /><div className="relation-mark"><ArrowRight size={19} /><span>potential association</span></div><EntitySelector label="Disease" entity={disease} options={diseases} onChange={setDisease} icon={<Dna size={17} />} /></div><button className="primary analyze" disabled={!compound || !disease || transitioning || !inferenceReady} onClick={() => compound && disease && onAnalyze(compound, disease)}>{transitioning ? 'Activating graph…' : inferenceReady ? 'Run hypothesis' : 'Inference unavailable'} <Sparkles size={16} /></button></section>
    {prediction && <PredictionSummary prediction={prediction} />}
    <NarrativeSections />
    <section className="final-cta"><div className="eyebrow">Begin a traceable investigation</div><h2>Explore<br /><em>the graph.</em></h2><button className="primary" onClick={() => document.getElementById('workspace')?.scrollIntoView({ behavior: 'smooth' })}>Start discovery <ArrowRight size={17} /></button></section>
  </>
}

function EntitySelector({ label, entity, options, onChange, icon }: { label: string; entity: Entity | null; options: Entity[]; onChange: (entity: Entity) => void; icon: ReactNode }) {
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const filtered = useMemo(() => options.filter(option => option.name.toLowerCase().includes(query.toLowerCase())).slice(0, 8), [options, query])
  return <div className="selector"><label>{icon}{label}</label><button className="selector-input" onClick={() => setOpen(!open)}><span>{entity?.name || `Search ${label.toLowerCase()}...`}</span><Search size={16} /></button>{open && <div className="selector-menu"><div className="search-field"><Search size={15} /><input autoFocus value={query} onChange={event => setQuery(event.target.value)} placeholder={`Search ${label.toLowerCase()}...`} /></div>{filtered.map(option => <button key={option.id} onClick={() => { onChange(option); setOpen(false); setQuery('') }}>{option.name}<span>{option.type}</span></button>)}</div>}</div>
}

function PredictionSummary({ prediction }: { prediction: Prediction }) {
  return <section className="result-card"><div className="result-header"><div><div className="eyebrow">Model result</div><h2>Potential relationship identified</h2></div><span className="score-type">MODEL SCORE</span></div><div className="result-entities"><div><span>COMPOUND</span><strong>{prediction.compound.name}</strong></div><div className="result-arrow"><ArrowRight size={20} /><small>treats</small></div><div><span>DISEASE</span><strong>{prediction.disease.name}</strong></div></div><div className="score-block"><div><span>Ranking score</span><strong>{prediction.score.toFixed(3)}</strong></div><div className="score-bar"><i style={{ width: `${prediction.score * 100}%` }} /></div><div className="result-meta"><span>{prediction.evidence_path_count} evidence paths</span><span>{prediction.score_type.replace('_', ' ')}</span></div></div><p className="scientific-note">{prediction.scientific_notice}</p></section>
}

function GraphPage({ prediction, onBack }: { prediction: Prediction | null; onBack: () => void }) {
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  if (!prediction) return <EmptyPage title="No evidence loaded" detail="Analyze a compound–disease pair first to retrieve its graph evidence." action={onBack} />
  const graph = prediction.evidence_graph
  const selected = graph.nodes.find(node => node.id === selectedNode)
  return <section className="graph-page"><div className="evidence-field"><div className="eyebrow">Knowledge representation · selected evidence neighborhood</div><KnowledgeField graph={graph} selectedId={selectedNode} /></div><div className="page-intro"><div><div className="eyebrow">Evidence graph</div><h1>{prediction.compound.name} <span>×</span> {prediction.disease.name}</h1><p>Actual metapath traces returned by the inference API. Pan, zoom, filter, highlight a path, or select a node to inspect its graph role.</p></div><button className="ghost" onClick={onBack}>← New hypothesis</button></div><div className="graph-layout"><div className="graph-panel"><EvidenceGraphExplorer graph={graph} selectedNode={selectedNode} onSelectNode={setSelectedNode} /></div><aside className="inspector"><div className="eyebrow">Node inspector</div>{selected ? <><div className={`node-chip ${selected.type.toLowerCase()}`}>{selected.type}</div><h2>{selected.name}</h2><div className="inspector-line"><span>Graph index</span><strong>{selected.index}</strong></div><div className="inspector-line"><span>Path positions</span><strong>{selected.path_positions.join(', ')}</strong></div><div className="inspector-line"><span>Connected evidence</span><strong>{graph.edges.filter(edge => edge.source === selected.id || edge.target === selected.id).length} relationships</strong></div></> : <div className="inspector-empty"><CircleHelp size={24} /><p>Click a node to inspect its type, graph index, and relationships.</p></div>}<div className="inspector-divider" /><div className="eyebrow">Interpretation</div><p className="small-copy">The graph shows structural evidence from deterministic metapath traversal. It is not literature retrieval and does not establish clinical efficacy.</p></aside></div><div className="path-list"><div className="section-heading"><div><div className="eyebrow">Traceable evidence</div><h2>Returned paths</h2></div><span className="mono">{graph.paths.length} PATHS · {graph.nodes.length} NODES</span></div>{graph.paths.length ? graph.paths.map(path => <div className="path-row" key={path.id}><span className="path-index">{path.id.replace('path-', '').padStart(2, '0')}</span><div><strong>{path.hops} hops</strong><span>{path.metapath}</span></div><ChevronRight size={17} /></div>) : <div className="empty-inline">The API returned the selected entities but no short deterministic metapath. The visualized edge is the requested inference target, not an observed biological fact.</div>}</div></section>
}

function Candidates({ compounds, diseases, response, setResponse, onInspect, inferenceReady }: { compounds: Entity[]; diseases: Entity[]; response: CandidateResponse | null; setResponse: (response: CandidateResponse) => void; onInspect: (candidate: Candidate, response: CandidateResponse) => void; inferenceReady: boolean }) {
  const [direction, setDirection] = useState<'diseases' | 'compounds'>('diseases')
  const [entity, setEntity] = useState<Entity | null>(null)
  const [limit, setLimit] = useState(10)
  const [busy, setBusy] = useState(false)
  const options = direction === 'diseases' ? compounds : diseases
  useEffect(() => { setEntity(options[0] || null) }, [direction, options.length])
  const run = async () => { if (!entity || !inferenceReady) return; setBusy(true); try { setResponse(await api.candidates(direction, entity.id, limit)) } finally { setBusy(false) } }
  return <section className="candidates-page"><div className="page-intro"><div><div className="eyebrow">Candidate discovery</div><h1>Explore therapeutic hypotheses.</h1><p>Rank graph entities using the cached decoder, then inspect any candidate's returned evidence graph.</p></div><div className="mode-switch"><button className={direction === 'diseases' ? 'active' : ''} onClick={() => setDirection('diseases')}>Drug → diseases</button><button className={direction === 'compounds' ? 'active' : ''} onClick={() => setDirection('compounds')}>Disease → compounds</button></div></div><div className="candidate-controls"><EntitySelector label={direction === 'diseases' ? 'Compound' : 'Disease'} entity={entity} options={options} onChange={setEntity} icon={direction === 'diseases' ? <Beaker size={17} /> : <Dna size={17} />} /><label className="limit-control">Top candidates<select value={limit} onChange={event => setLimit(Number(event.target.value))}><option value="5">5</option><option value="10">10</option><option value="20">20</option></select></label><button className="primary" onClick={run} disabled={busy || !entity}>{busy ? 'Ranking…' : 'Find candidates'} <ArrowRight size={16} /></button></div>{response ? <><div className="notice-line"><ShieldCheck size={16} />{response.scientific_notice}</div><div className="candidate-list">{response.candidates.map(candidate => <button className="candidate-row" key={candidate.entity.id} onClick={() => onInspect(candidate, response)}><span className="candidate-rank">{String(candidate.rank).padStart(2, '0')}</span><span className="candidate-name"><strong>{candidate.entity.name}</strong><small>{candidate.entity.type} · {candidate.evidence_path_count} evidence paths</small></span><span className="candidate-score"><strong>{candidate.score.toFixed(4)}</strong><small>MODEL SCORE</small></span><ChevronRight size={18} /></button>)}</div></> : <div className="empty-state"><Layers3 size={30} /><h2>Build a ranked research queue.</h2><p>Select an entity to compare candidate relationships.</p></div>}</section>
}

function MethodologyPage({ methodology, status }: { methodology: Methodology | null; status: Status | null }) {
  return <section className="methodology-page"><div className="page-intro"><div><div className="eyebrow">Model observatory</div><h1>How the hypothesis engine reasons.</h1><p>Implementation details pulled from the actual backend contract, not marketing claims.</p></div><div className="observatory-mark"><FlaskConical size={28} /><span>LOCAL INFERENCE</span></div></div><div className="pipeline">{methodology?.stages.map((stage, index) => <div className="stage" key={stage.title}><span>{String(index + 1).padStart(2, '0')}</span><h3>{stage.title}</h3><p>{stage.detail}</p>{index < (methodology.stages.length - 1) && <ArrowRight className="stage-arrow" size={18} />}</div>)}</div><div className="observatory-grid"><section className="technical-panel"><div className="eyebrow">Architecture</div><h2>Model observatory</h2>{methodology && Object.entries(methodology.model).map(([key, value]) => <div className="technical-row" key={key}><span>{key.replace('_', ' ')}</span><strong>{value}</strong></div>)}</section><section className="technical-panel"><div className="eyebrow">Runtime state</div><h2>System truth</h2><div className="technical-row"><span>Graph</span><strong>{status?.graph_label}</strong></div><div className="technical-row"><span>Checkpoint</span><strong>{status?.checkpoint}</strong></div><div className="technical-row"><span>Embeddings</span><strong>{status?.embeddings_cached ? 'Cached at startup' : 'Unavailable'}</strong></div><div className="technical-row"><span>External APIs</span><strong>None configured</strong></div></section></div><div className="caveat-panel"><CircleHelp size={19} /> <div><strong>Scientific boundary</strong><p>Graph evidence is deterministic metapath traversal, not PubMed retrieval or LLM-generated literature synthesis. Scores are ranking outputs. Any hypothesis requires validation outside this prototype.</p></div></div></section>
}

function HeroNetwork({ energized = false }: { energized?: boolean }) { return <div className={`hero-network ${energized ? 'is-energized' : ''}`}><div className="network-caption"><span className="pulse ready" /> {energized ? 'INFERENCE TARGET ACTIVATING' : 'KNOWLEDGE GRAPH REPRESENTATION'}</div><KnowledgeField compact energized={energized} /></div> }
function NarrativeSections() { return <div className="story" id="story"><section className="story-section split"><motion.div initial={{ opacity: 0, y: 28 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: .35 }}><div className="eyebrow">01 · The problem</div><h2>Biology is not a list.</h2></motion.div><p>Compounds, genes, pathways, and diseases continuously influence one another. Their most useful signals often live in the relationships between them, not in isolated records.</p></section><section className="story-section dark-stage"><div className="story-field"><KnowledgeField /></div><motion.div initial={{ opacity: 0, y: 28 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: .3 }}><div className="eyebrow">02 · The knowledge graph</div><h2>Everything is connected.</h2><p>Enter a representative biomedical field: gold compounds, cream genes, green pathways, and orange diseases. The hero is conceptual; analysis uses only the actual graph returned by the backend.</p><div className="story-stat"><strong>4</strong><span>biomedical entity types</span></div></motion.div></section><section className="story-section split engine-story"><div><div className="eyebrow">03 · The engine</div><h2>Let the graph reason.</h2><p>A compound's signal propagates through typed biomedical relationships before the HGT encoder and hybrid decoder rank a possible disease connection.</p><small className="concept-note">Conceptual visualization of graph message passing.</small></div><div className="engine-steps">{['Hetionet', 'Heterogeneous graph', '3 × HGT layers', 'Cached embeddings', 'Hybrid decoder', 'Hypothesis'].map((item, index) => <motion.div key={item} initial={{ opacity: 0, x: 22 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} transition={{ delay: index * .06 }}><span>{String(index + 1).padStart(2, '0')}</span><strong>{item}</strong><ArrowRight size={16} /></motion.div>)}</div></section></div> }
function LoadingState() { return <motion.div className="loading-state" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: .6 }}><Activity size={24} /><p>Initializing inference workstation…</p></motion.div> }
function EmptyPage({ title, detail, action }: { title: string; detail: string; action: () => void }) { return <div className="empty-state page-empty"><CircleHelp size={30} /><h2>{title}</h2><p>{detail}</p><button className="primary" onClick={action}>Return to discovery</button></div> }
function Footer({ demo }: { demo: boolean }) { return <footer><span>Research prototype · model-generated hypotheses require experimental validation.</span><span>{demo ? 'Demo graph active · no external evidence sources' : 'Graph evidence · local inference'}</span></footer> }

export default App
