import { Box, Flex, HStack, Link, Text } from '@chakra-ui/react'
import { NavLink, Outlet } from 'react-router-dom'

export default function Layout() {
  return (
    <Flex minH="100vh" bg="gray.100" direction="column">
      <Box px={8} py={4} bg="white" borderBottomWidth="1px" borderColor="gray.200">
        <Flex align="center" justify="space-between">
          <Text fontSize="md" fontWeight="800" color="blue.600" letterSpacing="tight">
            DrugOracle
          </Text>
          <HStack spacing={2}>
            <Link
              as={NavLink}
              to="/generate"
              px={3}
              py={1.5}
              borderRadius="md"
              fontSize="sm"
              fontWeight="600"
              color="gray.700"
              _activeLink={{ bg: 'gray.100', color: 'gray.900' }}
            >
              Generate
            </Link>
            <Link
              as={NavLink}
              to="/about"
              px={3}
              py={1.5}
              borderRadius="md"
              fontSize="sm"
              fontWeight="600"
              color="gray.700"
              _activeLink={{ bg: 'gray.100', color: 'gray.900' }}
            >
              About
            </Link>
          </HStack>
        </Flex>
      </Box>
      <Box as="main" flex={1} p={8} overflow="auto" bg="gray.50">
        <Outlet />
      </Box>
    </Flex>
  )
}
