import { Box, Heading, Text } from '@chakra-ui/react'

type PageHeaderProps = {
  title: string
  description: string
}

export function PageHeader({ title, description }: PageHeaderProps) {
  return (
    <Box mb={8} pb={4} borderBottomWidth="2px" borderBottomColor="blue.100">
      <Heading size="lg" fontWeight="700" color="gray.800" mb={2}>
        {title}
      </Heading>
      <Text fontSize="sm" color="gray.600">
        {description}
      </Text>
    </Box>
  )
}
