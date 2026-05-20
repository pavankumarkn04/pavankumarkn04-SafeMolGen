import {
  Box,
  Card,
  CardBody,
  Divider,
  Heading,
  ListItem,
  SimpleGrid,
  Text,
  UnorderedList,
} from '@chakra-ui/react'
import { PageHeader } from '../components/PageHeader'

type OptionDoc = {
  name: string
  what: string
  when: string
  watchouts: string
}

function OptionCard({ option }: { option: OptionDoc }) {
  return (
    <Card borderWidth="1px" borderColor="gray.200" shadow="sm" borderRadius="lg">
      <CardBody>
        <Heading as="h3" size="sm" mb={2}>
          {option.name}
        </Heading>
        <Text fontSize="sm" color="gray.700" mb={2}>
          <strong>What it does:</strong> {option.what}
        </Text>
        <Text fontSize="sm" color="gray.700" mb={2}>
          <strong>When to use:</strong> {option.when}
        </Text>
        <Text fontSize="sm" color="gray.700">
          <strong>Watchouts:</strong> {option.watchouts}
        </Text>
      </CardBody>
    </Card>
  )
}

const coreOptions: OptionDoc[] = [
  {
    name: 'Target success',
    what: 'Sets the clinical quality threshold the loop tries to reach. Higher values force stricter optimization.',
    when: 'Use 65-75% for balanced exploration; raise toward 80-90% for stricter design quality targets.',
    watchouts: 'Very high targets can increase runtime and may yield fewer valid converged runs.',
  },
  {
    name: 'Max iterations',
    what: 'Caps the number of generator-evaluator rounds in a run.',
    when: 'Use 8-12 for quick feedback, 15-25 for harder optimization tasks.',
    watchouts: 'Too low can stop before convergence; too high can spend time on diminishing returns.',
  },
  {
    name: 'Seed SMILES',
    what: 'Provides an initial scaffold to steer generation around a known chemistry region.',
    when: 'Use when you already have a lead scaffold and want focused optimization around it.',
    watchouts: 'Bad or narrow seeds can reduce diversity and trap search in local neighborhoods.',
  },
]

const propertyOptions: OptionDoc[] = [
  {
    name: 'logP range',
    what: 'Constrains lipophilicity window.',
    when: 'Use tighter windows when permeability or solubility balance is critical.',
    watchouts: 'Overly tight bounds can filter out most candidates early.',
  },
  {
    name: 'MW range',
    what: 'Constrains molecular weight in Daltons.',
    when: 'Use for target classes with known size expectations or delivery constraints.',
    watchouts: 'Too narrow a range reduces exploration and may block promising chemistry.',
  },
  {
    name: 'HBD max / HBA max',
    what: 'Limits hydrogen bond donor/acceptor counts.',
    when: 'Use to keep compounds in preferred oral-like property space.',
    watchouts: 'Hard cutoffs may reject potent motifs if limits are too aggressive.',
  },
  {
    name: 'TPSA max',
    what: 'Sets upper bound on topological polar surface area.',
    when: 'Use to tune permeability profile and avoid overly polar compounds.',
    watchouts: 'Very low TPSA caps can over-bias toward lipophilic structures.',
  },
  {
    name: 'QED min',
    what: 'Defines minimum acceptable drug-likeness score.',
    when: 'Use when filtering for practical medicinal chemistry quality early.',
    watchouts: 'High QED floor can remove unconventional but interesting candidates.',
  },
]

const advancedOptions: OptionDoc[] = [
  {
    name: 'Top-K',
    what: 'Controls sampling breadth at each token step. Lower K is conservative; higher K is exploratory.',
    when: '30-40 for stable runs, 50-80 when you need novelty and broader exploration.',
    watchouts: 'High K increases noise and invalid candidate risk.',
  },
  {
    name: 'Design mode: Single',
    what: 'Runs one continuous optimization trajectory.',
    when: 'Best default for speed and straightforward iteration.',
    watchouts: 'Can get stuck in local optima in difficult searches.',
  },
  {
    name: 'Design mode: Restarts',
    what: 'Runs multiple independent attempts and keeps the strongest result.',
    when: 'Use when single mode plateaus or results vary widely between runs.',
    watchouts: 'Runtime scales with restart count.',
  },
  {
    name: 'Design mode: Evolutionary',
    what: 'Uses broader iterative search behavior for diversity and exploration.',
    when: 'Use for novelty-heavy campaigns or escaping repeated local minima.',
    watchouts: 'May require more compute and tuning before stable convergence.',
  },
  {
    name: 'Restarts',
    what: 'Number of independent attempts in restart mode.',
    when: '3-5 for normal use, 6-10 when quality is more important than speed.',
    watchouts: 'More restarts increase total runtime linearly.',
  },
  {
    name: 'First-iter temp',
    what: 'Initial sampling temperature controlling early randomness.',
    when: '1.2-1.6 for exploration; 0.9-1.1 for more deterministic runs.',
    watchouts: 'Too high can produce unstable noisy candidates; too low may limit novelty.',
  },
  {
    name: 'Use RL model',
    what: 'Switches to RL-finetuned generator checkpoint for stronger objective alignment.',
    when: 'Use when target-driven optimization quality matters more than broad novelty.',
    watchouts: 'Can reduce diversity if RL checkpoint is narrowly tuned.',
  },
  {
    name: 'Improvement pacing',
    what: 'Applies progressive pressure across iterations to improve convergence behavior.',
    when: 'Keep ON for most runs; disable only when doing unconstrained exploration studies.',
    watchouts: 'When OFF, progression can be less stable and less directed.',
  },
]

export default function About() {
  return (
    <Box maxW="1400px" mx="auto">
      <PageHeader
        title="About and option guide"
        description="How each control changes generator-evaluator behavior, when to use each setting, and practical trade-offs."
      />

      <Card borderWidth="1px" borderColor="gray.200" shadow="sm" borderRadius="lg" mb={6}>
        <CardBody>
          <Heading as="h2" size="sm" mb={2}>
            How the loop works
          </Heading>
          <Text fontSize="sm" color="gray.700">
            Each iteration generates candidate molecules, evaluates them with ADMET + DrugOracle signals, ranks outcomes, and
            feeds feedback into the next iteration. Controls below shape exploration width, optimization pressure, and
            property constraints.
          </Text>
        </CardBody>
      </Card>

      <Heading as="h2" size="sm" mb={3}>
        Core controls
      </Heading>
      <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={4} mb={6}>
        {coreOptions.map((option) => (
          <OptionCard key={option.name} option={option} />
        ))}
      </SimpleGrid>

      <Divider mb={6} />

      <Heading as="h2" size="sm" mb={3}>
        Property targets
      </Heading>
      <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={4} mb={6}>
        {propertyOptions.map((option) => (
          <OptionCard key={option.name} option={option} />
        ))}
      </SimpleGrid>

      <Divider mb={6} />

      <Heading as="h2" size="sm" mb={3}>
        Advanced controls
      </Heading>
      <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={4} mb={6}>
        {advancedOptions.map((option) => (
          <OptionCard key={option.name} option={option} />
        ))}
      </SimpleGrid>

      <Card borderWidth="1px" borderColor="gray.200" shadow="sm" borderRadius="lg">
        <CardBody>
          <Heading as="h2" size="sm" mb={2}>
            Recommended presets
          </Heading>
          <UnorderedList fontSize="sm" color="gray.700" spacing={1.5} pl={4}>
            <ListItem>
              <strong>Balanced default:</strong> Top-K 40, mode Single, first-iter temp 1.4, RL OFF, pacing ON.
            </ListItem>
            <ListItem>
              <strong>Quality-focused:</strong> Top-K 30-40, mode Restarts (5), temp 1.1-1.3, RL ON, pacing ON.
            </ListItem>
            <ListItem>
              <strong>Exploration-focused:</strong> Top-K 60-80, mode Evolutionary, temp 1.5-1.8, RL OFF, pacing optional.
            </ListItem>
          </UnorderedList>
        </CardBody>
      </Card>
    </Box>
  )
}
