import { Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import About from './pages/About'
import Generate from './pages/Generate'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Navigate to="/generate" replace />} />
        <Route path="generate" element={<Generate />} />
        <Route path="about" element={<About />} />
      </Route>
    </Routes>
  )
}

export default App
