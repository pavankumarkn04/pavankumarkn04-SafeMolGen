import { useEffect, useRef, useState } from 'react'
import {
  Box,
  Button,
  Card,
  CardBody,
  Collapse,
  Flex,
  FormControl,
  FormLabel,
  Grid,
  Input,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Select,
  SimpleGrid,
  Slider,
  SliderFilledTrack,
  SliderThumb,
  SliderTrack,
  Switch,
  Text,
  useDisclosure,
  useToast,
  Spinner,
  Tooltip,
} from '@chakra-ui/react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts'
import { designStream, getConfig, moleculeSvgUrl, type ConfigResponse, type DesignParams, type DesignResult, getHealth } from '../api/client'
import { RangeSlider } from '../components/RangeSlider'
import { PageHeader } from '../components/PageHeader'
import { SectionHeader } from '../components/SectionHeader'

function ParamLabel({ label, help }: { label: string; help: string }) {
  return (
    <Flex align="center" gap={1.5} mb={1}>
      <FormLabel fontSize="xs" fontWeight="600" color="gray.700" m={0}>
        {label}
      </FormLabel>
      <Tooltip label={help} hasArrow fontSize="xs" placement="top-start">
        <Box
          as="button"
          type="button"
          w="16px"
          h="16px"
          borderRadius="full"
          borderWidth="1px"
          borderColor="gray.300"
          color="gray.500"
          fontSize="10px"
          lineHeight="14px"
          textAlign="center"
          bg="white"
          _hover={{ bg: 'gray.50', borderColor: 'gray.400' }}
          aria-label={`Info: ${label}`}
        >
          i
        </Box>
      </Tooltip>
    </Flex>
  )
}

export default function Generate() {
  const [generatorReady, setGeneratorReady] = useState<boolean | null>(null)
  const [modelsReady, setModelsReady] = useState<boolean | null>(null)
  const [config, setConfig] = useState<ConfigResponse | null>(null)
  const [targetSuccess, setTargetSuccess] = useState(0.3)
  const [maxIterations, setMaxIterations] = useState(10)
  const [seedSmiles, setSeedSmiles] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<DesignResult | null>(null)
  const [streamingHistory, setStreamingHistory] = useState<DesignResult[]>([])
  const [streamStarted, setStreamStarted] = useState(false)
  const { isOpen: propsOpen, onToggle: toggleProps } = useDisclosure({ defaultIsOpen: false })
  const { isOpen: propTargetsOpen, onToggle: togglePropTargets } = useDisclosure({ defaultIsOpen: true })
  const [logpMin, setLogpMin] = useState(2)
  const [logpMax, setLogpMax] = useState(5)
  const [mwMin, setMwMin] = useState(150)
  const [mwMax, setMwMax] = useState(500)
  const [hbdMax, setHbdMax] = useState(5)
  const [hbaMax, setHbaMax] = useState(10)
  const [tpsaMax, setTpsaMax] = useState(140)
  const [qedMin, setQedMin] = useState(0.5)
  const [topK, setTopK] = useState(40)
  const [useRlModel, setUseRlModel] = useState(false)
  const [designMode, setDesignMode] = useState<'single' | 'restarts' | 'evolutionary'>('single')
  const [nRestarts, setNRestarts] = useState(5)
  const [firstIterationTemperature, setFirstIterationTemperature] = useState<string>('')
  const [useImprovementPacing, setUseImprovementPacing] = useState(true)
  const [selectionMode, setSelectionMode] = useState('phase_weighted')
  const [diversityTanimotoMax, setDiversityTanimotoMax] = useState(0.7)
  const [explorationFraction, setExplorationFraction] = useState(0.25)
  const [maxRotatableBonds, setMaxRotatableBonds] = useState(15)
  const [safetyThreshold, setSafetyThreshold] = useState(0.2)
  const [requireNoAlerts, setRequireNoAlerts] = useState(false)
  const [useOracleFeedback, setUseOracleFeedback] = useState(true)
  const abortControllerRef = useRef<AbortController | null>(null)

  const toast = useToast()

  useEffect(() => {
    getConfig().then((c) => {
      setConfig(c)
      if (c.diversity_tanimoto_max_default != null) setDiversityTanimotoMax(c.diversity_tanimoto_max_default)
      if (c.exploration_fraction_default != null) setExplorationFraction(c.exploration_fraction_default)
      if (c.max_rotatable_bonds_default != null) setMaxRotatableBonds(c.max_rotatable_bonds_default)
    }).catch(() => setConfig(null))
    getHealth().then((h) => {
      setGeneratorReady(h.generator_loaded)
      setModelsReady(h.models_loaded ?? false)
    }).catch(() => {
      setGeneratorReady(false)
      setModelsReady(false)
    })
  }, [])

  const firstIterTempNum = firstIterationTemperature === '' ? (config?.first_iteration_temperature_default ?? 1.4) : Number(firstIterationTemperature)
  const firstIterTempValue = Number.isFinite(firstIterTempNum) ? firstIterTempNum : (config?.first_iteration_temperature_default ?? 1.4)

  const propertyTargets: Record<string, number | [number, number]> = {
    logp: [logpMin, logpMax],
    mw_min: mwMin,
    mw: mwMax,
    hbd: hbdMax,
    hba: hbaMax,
    tpsa: tpsaMax,
    qed: qedMin,
  }

  const selectionModeOptions = config?.selection_modes?.length
    ? config.selection_modes
    : ['overall', 'pareto', 'diversity', 'phase_weighted', 'bottleneck']

  const params: DesignParams = {
    target_success: targetSuccess,
    max_iterations: maxIterations,
    candidates_per_iteration: 220,
    top_k: topK,
    property_targets: propertyTargets,
    seed_smiles: seedSmiles.trim() || undefined,
    use_rl_model: useRlModel,
    design_mode: designMode,
    n_restarts: designMode === 'restarts' ? nRestarts : undefined,
    use_improvement_pacing: useImprovementPacing,
    safety_threshold: safetyThreshold,
    require_no_structural_alerts: requireNoAlerts,
    use_oracle_feedback: useOracleFeedback,
    selection_mode: selectionMode,
    diversity_tanimoto_max: diversityTanimotoMax,
    exploration_fraction: explorationFraction,
    max_rotatable_bonds: maxRotatableBonds,
    ...(firstIterationTemperature !== '' && Number.isFinite(Number(firstIterationTemperature)) && { first_iteration_temperature: Number(firstIterationTemperature) }),
  }

  function handleRun() {
    abortControllerRef.current?.abort()
    abortControllerRef.current = new AbortController()
    setLoading(true)
    setResult(null)
    setStreamingHistory([])
    setStreamStarted(false)
    designStream(
      params,
      (data) => {
        setStreamingHistory((prev) => [...prev, data])
        setResult(data)
      },
      (data) => {
        setResult(data)
        setLoading(false)
        toast({ title: data.target_achieved ? 'Target achieved' : 'Run finished', status: 'success', isClosable: true })
      },
      (err) => {
        setLoading(false)
        if (err === 'Cancelled') {
          setResult(null)
          setStreamingHistory([])
          toast({ title: 'Generation stopped', status: 'info', isClosable: true })
        } else {
          toast({ title: 'Error', description: err, status: 'error', isClosable: true })
        }
      },
      { signal: abortControllerRef.current.signal, onStarted: () => setStreamStarted(true) }
    )
  }

  function handleStop() {
    abortControllerRef.current?.abort()
  }

  const displayResult = result ?? (streamingHistory.length ? streamingHistory[streamingHistory.length - 1] : null)
  const chartData = (displayResult?.history ?? []).map((h) => ({
    iteration: h.iteration,
    overall: Math.round(h.overall_prob * 10000) / 100,
    phase1: Math.round(h.phase1_prob * 10000) / 100,
    phase2: Math.round(h.phase2_prob * 10000) / 100,
    phase3: Math.round(h.phase3_prob * 10000) / 100,
  }))

  return (
    <Box maxW="1600px" mx="auto">
      <PageHeader
        title="Molecule generation"
        description="Set targets and run AI-driven design loop. Best molecule and optimization curve update as results stream."
      />

      <Grid templateColumns={{ base: '1fr', xl: '280px 1fr' }} gap={6} alignItems="start">
        <Card borderWidth="1px" borderColor="gray.200" shadow="sm" borderRadius="lg">
          <CardBody py={4} px={4}>
            <FormControl mb={3}>
              <ParamLabel label="Target success" help="Clinical quality threshold to stop optimization early." />
              <Slider value={targetSuccess} min={0.1} max={0.95} step={0.05} onChange={setTargetSuccess} size="sm">
                <SliderTrack><SliderFilledTrack /></SliderTrack>
                <SliderThumb />
              </Slider>
              <Text fontSize="xs" color="gray.500">{Math.round(targetSuccess * 100)}%</Text>
            </FormControl>
            <FormControl mb={3}>
              <ParamLabel label="Max iterations" help="Upper bound on generator-evaluator loop rounds." />
              <NumberInput value={maxIterations} min={1} max={50} size="sm" onChange={(_, v) => setMaxIterations(v || 10)}>
                <NumberInputField />
                <NumberInputStepper>
                  <NumberIncrementStepper />
                  <NumberDecrementStepper />
                </NumberInputStepper>
              </NumberInput>
            </FormControl>
            <FormControl mb={3}>
              <ParamLabel label="Seed SMILES" help="Optional starting scaffold to steer generation." />
              <Input size="sm" value={seedSmiles} onChange={(e) => setSeedSmiles(e.target.value)} placeholder="Optional scaffold" />
            </FormControl>

            <Button size="sm" variant="ghost" colorScheme="gray" onClick={togglePropTargets} mb={2}>
              {propTargetsOpen ? '−' : '+'} Property targets
            </Button>
            <Collapse in={propTargetsOpen}>
              <Box mb={3}>
                <RangeSlider
                  label="logP"
                  value={[logpMin, logpMax]}
                  min={-2}
                  max={10}
                  step={0.5}
                  onChange={([a, b]) => { setLogpMin(a); setLogpMax(b) }}
                  helperText="Lipophilicity; drug-like often 2–5."
                  infoText="Partition coefficient target range."
                />
                <RangeSlider
                  label="MW (Da)"
                  value={[mwMin, mwMax]}
                  min={0}
                  max={800}
                  step={10}
                  onChange={([a, b]) => { setMwMin(a); setMwMax(b) }}
                  helperText="Molecular weight range (e.g. 150–500)."
                  infoText="Allowed molecular weight window in Daltons."
                />
                <FormControl mb={2}>
                  <ParamLabel label={`HBD max — ${hbdMax}`} help="Maximum hydrogen bond donors." />
                  <Slider value={hbdMax} min={0} max={15} step={1} onChange={setHbdMax} size="sm"><SliderTrack><SliderFilledTrack /></SliderTrack><SliderThumb /></Slider>
                </FormControl>
                <FormControl mb={2}>
                  <ParamLabel label={`HBA max — ${hbaMax}`} help="Maximum hydrogen bond acceptors." />
                  <Slider value={hbaMax} min={0} max={20} step={1} onChange={setHbaMax} size="sm"><SliderTrack><SliderFilledTrack /></SliderTrack><SliderThumb /></Slider>
                </FormControl>
                <FormControl mb={2}>
                  <ParamLabel label={`TPSA max — ${tpsaMax}`} help="Topological polar surface area upper bound." />
                  <Slider value={tpsaMax} min={0} max={200} step={5} onChange={setTpsaMax} size="sm"><SliderTrack><SliderFilledTrack /></SliderTrack><SliderThumb /></Slider>
                </FormControl>
                <FormControl mb={2}>
                  <ParamLabel label={`QED min — ${qedMin}`} help="Minimum quantitative estimate of drug-likeness." />
                  <Slider value={qedMin} min={0} max={1} step={0.05} onChange={setQedMin} size="sm"><SliderTrack><SliderFilledTrack /></SliderTrack><SliderThumb /></Slider>
                </FormControl>
              </Box>
            </Collapse>

            <Button size="sm" variant="ghost" colorScheme="gray" onClick={toggleProps} mb={2}>
              {propsOpen ? '−' : '+'} Advanced
            </Button>
            <Collapse in={propsOpen}>
              <Box mb={3}>
                <FormControl mb={2}>
                  <ParamLabel label="Top-K" help="Sampling breadth per token during candidate generation." />
                  <NumberInput size="sm" value={topK} min={1} max={80} step={1} onChange={(_, v) => setTopK(v ?? 40)}>
                    <NumberInputField />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                </FormControl>
                <FormControl mb={2}>
                  <ParamLabel label="Design mode" help="Single run, restart-based search, or evolutionary exploration." />
                  <Select size="sm" value={designMode} onChange={(e) => setDesignMode(e.target.value as 'single' | 'restarts' | 'evolutionary')}>
                    <option value="single">Single</option>
                    <option value="restarts">Restarts</option>
                    <option value="evolutionary">Evolutionary</option>
                  </Select>
                </FormControl>
                {designMode === 'restarts' && (
                  <FormControl mb={2}>
                    <ParamLabel label="Restarts" help="Number of fresh attempts in restart mode." />
                    <NumberInput size="sm" value={nRestarts} min={1} max={10} step={1} onChange={(_, v) => setNRestarts(v ?? 5)}>
                      <NumberInputField />
                      <NumberInputStepper>
                        <NumberIncrementStepper />
                        <NumberDecrementStepper />
                      </NumberInputStepper>
                    </NumberInput>
                  </FormControl>
                )}
                <FormControl mb={2}>
                  <ParamLabel label="First-iter temp" help="Initial sampling temperature before adaptive updates." />
                  <Input
                    size="sm"
                    type="text"
                    inputMode="decimal"
                    value={firstIterationTemperature === '' ? String(firstIterTempValue) : firstIterationTemperature}
                    onChange={(e) => setFirstIterationTemperature(e.target.value)}
                    placeholder="e.g. 1.4"
                  />
                </FormControl>
                <FormControl display="flex" alignItems="center" mb={1}>
                  <ParamLabel label="Use RL model" help="Switch to RL-finetuned generator checkpoint." />
                  <Switch size="sm" isChecked={useRlModel} onChange={(e) => setUseRlModel(e.target.checked)} />
                </FormControl>
                <FormControl display="flex" alignItems="center" mb={1}>
                  <ParamLabel label="Improvement pacing" help="Gradually tighten target pressure across iterations." />
                  <Switch size="sm" isChecked={useImprovementPacing} onChange={(e) => setUseImprovementPacing(e.target.checked)} />
                </FormControl>
                <FormControl mb={2}>
                  <ParamLabel label="Selection mode" help="How to pick the iteration best: phase-weighted (default), diversity vs current best, Pareto front, etc." />
                  <Select size="sm" value={selectionMode} onChange={(e) => setSelectionMode(e.target.value)}>
                    {selectionModeOptions.map((m) => (
                      <option key={m} value={m}>{m.replace(/_/g, ' ')}</option>
                    ))}
                  </Select>
                </FormControl>
                <FormControl mb={2}>
                  <ParamLabel label={`Diversity Tanimoto max — ${diversityTanimotoMax.toFixed(2)}`} help="When using diversity mode (or target+diversity): max fingerprint similarity to running best (lower = more structurally different)." />
                  <Slider value={diversityTanimotoMax} min={0.3} max={0.95} step={0.05} onChange={setDiversityTanimotoMax} size="sm">
                    <SliderTrack><SliderFilledTrack /></SliderTrack>
                    <SliderThumb />
                  </Slider>
                </FormControl>
                <FormControl mb={2}>
                  <ParamLabel label={`Exploration fraction — ${Math.round(explorationFraction * 100)}%`} help="Fraction of unconditioned high-temperature samples each iteration to escape plateaus." />
                  <Slider value={explorationFraction} min={0} max={0.5} step={0.05} onChange={setExplorationFraction} size="sm">
                    <SliderTrack><SliderFilledTrack /></SliderTrack>
                    <SliderThumb />
                  </Slider>
                </FormControl>
                <FormControl mb={2}>
                  <ParamLabel label="Max rotatable bonds" help="Reject candidates above this count (grease / flexible chains). Use -1 for no cap." />
                  <NumberInput size="sm" value={maxRotatableBonds} min={-1} max={30} step={1} onChange={(_, v) => setMaxRotatableBonds(v ?? 15)}>
                    <NumberInputField />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                </FormControl>
                <FormControl mb={2}>
                  <ParamLabel label={`Safety threshold — ${safetyThreshold.toFixed(2)}`} help="Minimum clinical quality to mark an iteration as passing safety." />
                  <Slider value={safetyThreshold} min={0.05} max={0.5} step={0.05} onChange={setSafetyThreshold} size="sm">
                    <SliderTrack><SliderFilledTrack /></SliderTrack>
                    <SliderThumb />
                  </Slider>
                </FormControl>
                <FormControl display="flex" alignItems="center" mb={1}>
                  <ParamLabel label="Require no structural alerts" help="Iterations only pass safety if no structural alert flags." />
                  <Switch size="sm" isChecked={requireNoAlerts} onChange={(e) => setRequireNoAlerts(e.target.checked)} />
                </FormControl>
                <FormControl display="flex" alignItems="center" mb={1}>
                  <ParamLabel label="Oracle steering" help="Use oracle feedback to condition the next batch (disable for one round when stuck is handled inside the pipeline)." />
                  <Switch size="sm" isChecked={useOracleFeedback} onChange={(e) => setUseOracleFeedback(e.target.checked)} />
                </FormControl>
              </Box>
            </Collapse>

            <Flex gap={2} mt={2} wrap="wrap">
              <Button colorScheme="blue" size="sm" flex={1} minW="120px" onClick={handleRun} isLoading={loading} loadingText="Running…" isDisabled={loading || modelsReady === false}>
                Run generation
              </Button>
              <Button size="sm" colorScheme="red" variant="outline" flex={1} minW="120px" onClick={handleStop} isDisabled={!loading}>
                Stop generation
              </Button>
            </Flex>
            {(generatorReady === false || modelsReady === false) && (
              <Text fontSize="xs" color="red.600" mt={2}>
                {generatorReady === false ? 'Generator not loaded (check checkpoints/generator).' : 'Evaluator models not loaded (check checkpoints/admet and checkpoints/oracle).'}
              </Text>
            )}
          </CardBody>
        </Card>

        <Box>
          <Grid templateColumns="1fr" gap={6} mb={6} alignItems="start">
            <Box minW={0}>
              <Card borderWidth="1px" borderColor="gray.200" shadow="sm" borderRadius="lg">
                <CardBody py={4} px={4}>
                  <SectionHeader>Best molecule</SectionHeader>
                  {displayResult?.final_smiles ? (
                    <>
                      <Box borderWidth="1px" borderColor="gray.200" borderRadius="lg" overflow="hidden" w="100%" maxW="260px" h="190px" bg="white" mx="auto">
                        <img src={moleculeSvgUrl(displayResult.final_smiles, 260, 190)} alt="Best molecule" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                      </Box>
                      <Text fontFamily="mono" fontSize="xs" wordBreak="break-all" mt={2} color="gray.700">
                        {displayResult.canonical_smiles ?? displayResult.final_smiles}
                      </Text>
                    </>
                  ) : (
                    <Box w="100%" maxW="260px" minH="190px" borderWidth="1px" borderStyle="dashed" borderColor="gray.300" borderRadius="lg" display="flex" flexDirection="column" alignItems="center" justifyContent="center" gap={2} bg="gray.50" mx="auto" p={3}>
                      {loading && <Spinner size="sm" color="blue.500" />}
                      <Text fontSize="xs" color="gray.500" textAlign="center">
                        {loading ? (streamStarted ? 'Pipeline started, generating…' : 'Generating…') : 'Molecule will appear here'}
                      </Text>
                    </Box>
                  )}
                </CardBody>
              </Card>

              {displayResult && (
                <Flex mt={3} gap={2} wrap="wrap">
                  <Box px={2} py={0.5} bg={displayResult.target_achieved ? 'green.100' : 'orange.100'} borderRadius="md">
                    <Text fontSize="xs" fontWeight="600" color={displayResult.target_achieved ? 'green.800' : 'orange.800'}>
                      {displayResult.target_achieved ? 'Target achieved' : 'Below target'}
                    </Text>
                  </Box>
                  {displayResult._strategy_used && (
                    <Box px={2} py={0.5} bg="gray.100" borderRadius="md">
                      <Text fontSize="xs" fontWeight="500" color="gray.700">Strategy: {displayResult._strategy_used}</Text>
                    </Box>
                  )}
                  <Box px={2} py={0.5} bg="gray.100" borderRadius="md">
                    <Text fontSize="xs" fontWeight="500" color="gray.700">Iterations: {displayResult.total_iterations}</Text>
                  </Box>
                </Flex>
              )}

              <Box mt={4}>
                <SectionHeader>Phase probabilities & clinical quality</SectionHeader>
                {displayResult ? (
                  <SimpleGrid columns={{ base: 2, sm: 4 }} spacing={3}>
                    {[
                      { label: 'Phase I', value: displayResult.final_phase1 * 100, color: 'green.600' },
                      { label: 'Phase II', value: displayResult.final_phase2 * 100, color: 'orange.600' },
                      { label: 'Phase III', value: displayResult.final_phase3 * 100, color: 'purple.600' },
                      { label: 'Clinical quality', value: displayResult.final_overall * 100, color: 'blue.600' },
                    ].map((it) => (
                      <Box key={it.label} p={3} bg="gray.50" borderRadius="md" borderWidth="1px" borderColor="gray.200" textAlign="center">
                        <Text fontSize="xs" color="gray.500">{it.label}</Text>
                        <Text fontSize="lg" fontWeight="700" color={it.color}>{it.value.toFixed(1)}%</Text>
                      </Box>
                    ))}
                  </SimpleGrid>
                ) : (
                  <Box p={4} bg="gray.50" borderRadius="md" borderWidth="1px" borderColor="gray.200">
                    <Text fontSize="sm" color="gray.500">Run generation to see phase probabilities.</Text>
                  </Box>
                )}
              </Box>

              <Box mt={4}>
                <SectionHeader>Optimization journey</SectionHeader>
                {chartData.length > 0 ? (
                  <Box h="280px" w="100%" bg="white" borderRadius="md" borderWidth="1px" borderColor="gray.200" p={3}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="gray.100" />
                        <XAxis dataKey="iteration" tick={{ fontSize: 11 }} />
                        <YAxis yAxisId="left" domain={[0, 100]} tick={{ fontSize: 11 }} width={32} />
                        <YAxis yAxisId="right" orientation="right" domain={[0, 100]} tick={{ fontSize: 11 }} width={32} />
                        <ReferenceLine yAxisId="right" y={targetSuccess * 100} stroke="gray" strokeDasharray="5 5" />
                        <RechartsTooltip
                          formatter={(value) => [typeof value === 'number' ? `${value.toFixed(1)}%` : String(value), '']}
                        />
                        <Legend wrapperStyle={{ fontSize: '11px' }} />
                        <Line yAxisId="right" type="monotone" dataKey="overall" name="Clinical quality" stroke="#3182CE" strokeWidth={2} dot={{ r: 2.5 }} />
                        <Line yAxisId="left" type="monotone" dataKey="phase1" name="Phase I" stroke="#38A169" strokeWidth={1.5} dot={{ r: 2 }} />
                        <Line yAxisId="left" type="monotone" dataKey="phase2" name="Phase II" stroke="#D69E2E" strokeWidth={1.5} dot={{ r: 2 }} />
                        <Line yAxisId="left" type="monotone" dataKey="phase3" name="Phase III" stroke="#805AD5" strokeWidth={1.5} dot={{ r: 2 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </Box>
                ) : (
                  <Box h="200px" bg="gray.50" borderRadius="md" borderWidth="1px" borderStyle="dashed" borderColor="gray.300" display="flex" alignItems="center" justifyContent="center">
                    <Text fontSize="sm" color="gray.500">Chart appears while design is running.</Text>
                  </Box>
                )}
              </Box>

              {displayResult?.history && displayResult.history.length > 0 && (
                <Box mt={4}>
                  <SectionHeader>Iteration details</SectionHeader>
                  <Box maxH="260px" overflowY="auto" borderWidth="1px" borderColor="gray.200" borderRadius="md" bg="white">
                    {displayResult.history.map((h, i) => (
                      <Box key={i} px={3} py={2} borderBottomWidth={i < displayResult.history.length - 1 ? '1px' : '0'} borderColor="gray.100">
                        <Flex justify="space-between" align="center">
                          <Text fontSize="xs" fontWeight="700" color="gray.700">Iteration {h.iteration}</Text>
                          <Flex gap={2}>
                            <Text fontSize="xs" color={h.passed_safety ? 'green.600' : 'red.500'}>{h.passed_safety ? 'Safe' : 'Unsafe'}</Text>
                            {h.used_oracle_feedback && <Text fontSize="10px" color="blue.500" fontWeight="500">Oracle-steered</Text>}
                          </Flex>
                        </Flex>
                        <Text fontSize="xs" color="gray.500" fontFamily="mono" noOfLines={1}>{h.smiles || '(no candidate)'}</Text>
                        <Text fontSize="xs" color="gray.600">
                          Overall: {(h.overall_prob * 100).toFixed(1)}% &middot; P1: {(h.phase1_prob * 100).toFixed(1)}% &middot; P2: {(h.phase2_prob * 100).toFixed(1)}% &middot; P3: {(h.phase3_prob * 100).toFixed(1)}%
                        </Text>
                        {h.improvements.length > 0 && (
                          <Text fontSize="xs" color="green.600">{h.improvements.join(' | ')}</Text>
                        )}
                        {h.structural_alerts.length > 0 && (
                          <Text fontSize="xs" color="red.500">Alerts: {h.structural_alerts.join(', ')}</Text>
                        )}
                      </Box>
                    ))}
                  </Box>
                </Box>
              )}

              {displayResult?.recommendations && displayResult.recommendations.length > 0 && (
                <Box mt={4}>
                  <SectionHeader>Recommendations</SectionHeader>
                  {displayResult.recommendations.map((rec, i) => {
                    const sev = (rec.severity ?? '').toLowerCase()
                    const borderColor = sev === 'high' ? 'red.400' : sev === 'medium' ? 'orange.400' : sev === 'positive' ? 'green.400' : 'gray.300'
                    const bg = sev === 'high' ? 'red.50' : sev === 'medium' ? 'orange.50' : sev === 'positive' ? 'green.50' : 'gray.50'
                    const typeColor = sev === 'high' ? 'red.700' : sev === 'medium' ? 'orange.700' : sev === 'positive' ? 'green.700' : 'gray.600'
                    return (
                      <Box key={i} py={2} px={3} mb={1.5} borderLeftWidth="3px" borderColor={borderColor} bg={bg} borderRadius="md">
                        <Flex justify="space-between" align="center" mb={0.5}>
                          <Text fontSize="xs" fontWeight="700" color={typeColor} textTransform="uppercase">{rec.type}</Text>
                          {rec.severity && <Text fontSize="10px" color={typeColor} fontWeight="600">{rec.severity}</Text>}
                        </Flex>
                        {rec.issue && <Text fontSize="sm" fontWeight="500" color="gray.800">{rec.issue}</Text>}
                        <Text fontSize="sm" color="gray.600">{rec.suggestion}</Text>
                        {rec.expected_improvement && <Text fontSize="xs" color="gray.500" mt={0.5}>Expected: {rec.expected_improvement}</Text>}
                      </Box>
                    )
                  })}
                </Box>
              )}
            </Box>
          </Grid>
        </Box>
      </Grid>
    </Box>
  )
}
