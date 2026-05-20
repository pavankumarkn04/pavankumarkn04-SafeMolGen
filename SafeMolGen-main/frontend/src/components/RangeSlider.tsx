import {
  Box,
  FormControl,
  FormHelperText,
  Flex,
  FormLabel,
  Slider,
  SliderFilledTrack,
  SliderThumb,
  SliderTrack,
  Text,
  Tooltip,
} from '@chakra-ui/react'

type RangeSliderProps = {
  label: string
  value: [number, number]
  min: number
  max: number
  step?: number
  onChange: (value: [number, number]) => void
  helperText?: string
  infoText?: string
}

export function RangeSlider({
  label,
  value: [lo, hi],
  min,
  max,
  step = 1,
  onChange,
  helperText,
  infoText,
}: RangeSliderProps) {
  const setLo = (v: number) => onChange([Math.min(v, hi), hi])
  const setHi = (v: number) => onChange([lo, Math.max(v, lo)])

  return (
    <FormControl>
      <FormLabel fontSize="sm" mb={1}>
        <Flex as="span" align="center" gap={1.5}>
          <Text as="span">{label}</Text>
          {infoText && (
            <Tooltip label={infoText} hasArrow placement="top" openDelay={180}>
              <Box
                as="span"
                w="16px"
                h="16px"
                borderRadius="full"
                borderWidth="1px"
                borderColor="gray.400"
                color="gray.500"
                fontSize="10px"
                lineHeight="14px"
                textAlign="center"
                fontWeight="700"
                cursor="help"
              >
                i
              </Box>
            </Tooltip>
          )}
          <Text as="span" fontWeight="normal" color="gray.500" ml={1}>
            {lo} – {hi}
          </Text>
        </Flex>
      </FormLabel>
      <Box position="relative" py={2}>
        <Box
          position="absolute"
          left={0}
          right={0}
          top="50%"
          transform="translateY(-50%)"
          h="6px"
          borderRadius="full"
          bg="gray.200"
          pointerEvents="none"
        />
        <Box position="relative" zIndex={1} h="24px">
          <Box position="absolute" left={0} right={0} top={0} bottom={0}>
            <Slider
              value={lo}
              min={min}
              max={max}
              step={step}
              onChange={setLo}
              size="sm"
              focusThumbOnChange={false}
              sx={{
                '& .chakra-slider__track': { bg: 'transparent', h: '6px' },
                '& .chakra-slider__filled-track': { bg: 'transparent' },
                '& .chakra-slider__thumb': { zIndex: 2 },
              }}
            >
              <SliderTrack><SliderFilledTrack /></SliderTrack>
              <SliderThumb />
            </Slider>
          </Box>
          <Box position="absolute" left={0} right={0} top={0} bottom={0}>
            <Slider
              value={hi}
              min={min}
              max={max}
              step={step}
              onChange={setHi}
              size="sm"
              focusThumbOnChange={false}
              sx={{
                '& .chakra-slider__track': { bg: 'transparent', h: '6px' },
                '& .chakra-slider__filled-track': { bg: 'transparent' },
                '& .chakra-slider__thumb': { zIndex: 2 },
              }}
            >
              <SliderTrack><SliderFilledTrack /></SliderTrack>
              <SliderThumb />
            </Slider>
          </Box>
        </Box>
        <Box
          position="absolute"
          left={`${((lo - min) / (max - min)) * 100}%`}
          right={`${((max - hi) / (max - min)) * 100}%`}
          top="50%"
          transform="translateY(-50%)"
          h="6px"
          borderRadius="full"
          bg="blue.400"
          pointerEvents="none"
          zIndex={0}
        />
      </Box>
      {helperText && (
        <FormHelperText fontSize="xs" mt={1} color="gray.500">
          {helperText}
        </FormHelperText>
      )}
    </FormControl>
  )
}
