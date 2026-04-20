import { Routes, Route, Navigate } from "react-router-dom"
import { AdminPage } from "./pages/AdminPage"
import { UserPage } from "./pages/UserPage"
import "./App.css"

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<UserPage />} />
      <Route path="/admin" element={<AdminPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}