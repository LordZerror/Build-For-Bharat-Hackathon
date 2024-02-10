import React from 'react'
import HomePage from './components/HomePage'
import Error from './components/Error'
import './CSS/app.css'
import { BrowserRouter, Routes, Route} from 'react-router-dom'
import ProductsLayout from './components/ProductsLayout'
import Cart from './components/Cart'
import Login from './components/Users/Login'
import Register from './components/Users/Register'
import OrderHistory from './components/Users/OrderHistory'

function App() {
  return (
      <BrowserRouter>
        <Routes>
            <Route  path="/" element={<HomePage />} />
            <Route  path="/:productId" element={<ProductsLayout />} />
            <Route  path="/My-Cart" element={<Cart />} />
            <Route  path="*" element={<Error/>} />
            <Route  path="/Login" element={<Login />} />
            <Route  path="/Register" element={<Register />} />
            <Route  path="/OrderHistory" element={<OrderHistory/>}/>
        </Routes>
      </BrowserRouter>
  );
}

export default App;
