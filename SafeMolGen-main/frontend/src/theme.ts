import { extendTheme } from '@chakra-ui/react'

const theme = extendTheme({
  components: {
    Card: {
      baseStyle: {
        container: {
          borderRadius: 'xl',
          shadow: 'md',
          borderWidth: '1px',
          borderColor: 'gray.200',
          bg: 'white',
        },
      },
    },
    Button: {
      baseStyle: {
        fontWeight: 'medium',
      },
    },
    Input: {
      defaultProps: {
        focusBorderColor: 'blue.500',
      },
    },
    Select: {
      defaultProps: {
        focusBorderColor: 'blue.500',
      },
    },
    NumberInput: {
      defaultProps: {
        focusBorderColor: 'blue.500',
      },
    },
  },
})

export default theme
