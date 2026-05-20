import { Box, Text } from '@chakra-ui/react'

type SectionHeaderProps = {
  children: React.ReactNode
}

export function SectionHeader({ children }: SectionHeaderProps) {
  return (
    <Box display="flex" alignItems="center" gap={2} mb={3}>
      <Box w="4px" h="5" borderRadius="full" bg="blue.500" flexShrink={0} />
      <Text
        fontSize="xs"
        fontWeight="700"
        color="gray.700"
        textTransform="uppercase"
        letterSpacing="wider"
      >
        {children}
      </Text>
    </Box>
  )
}
