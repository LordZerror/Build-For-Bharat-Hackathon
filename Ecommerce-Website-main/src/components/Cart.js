import React from 'react'
import '../CSS/cart.css'
import Navbar from './Navbar'
import { useStateContext } from '../context/StateContext'


function Cart(){

  const { totalPrice, totalQuantities, cartItems, toggleCartItemQuanitity, onRemove } = useStateContext();

  return (
    <div>
        <div>
            <Navbar />
        </div>
        <div className='shopping--details'>
          <div className='cart-items--quantity'>
            <div className='your--shopping--cart'><b><h3>Your Shopping Cart</h3></b></div>
            <div className='totalQuantities--label'><p>({totalQuantities} items)</p></div>
            <div >
              <h3 className='sub--total'>Subtotal:${totalPrice}</h3>
           </div>
           {cartItems.length > 0 && (
             <div>
               <button className="checkout-button">Checkout</button>
             </div>
           )}
          </div>
          <div className='background'>
             {cartItems.map((item)=>(
              <div className='product-block' key={item.id}>
              <div>
                <img className='product-thumbnail' src={item.thumbnail}></img> 
              </div>
              <div className='product-info'>
                <div className='product-details'>
                  <h3 className='product-title'>{item.title}</h3>
                  <p className='product-price'>${item.price}</p>
                </div>
                <div className='final--quantity'>
                  <div className='quantity-d'>
                    <p className='qty-label'>Qty :</p><button className='decrement-qty' onClick={() => toggleCartItemQuanitity(item.id, 'dec') }>-</button><span className='qty-counter'>{item.quantity}</span><button className='increment-qty' onClick={() => toggleCartItemQuanitity(item.id, 'inc') }>+</button>
                  </div>
                  <div>
                    <button type="button" className="remove-item" onClick={() => onRemove(item)}>
                      <b>DELETE</b>
                    </button>
                  </div>
                </div>
              </div>
            </div>
             ))}
          </div>
        </div>
    </div>
  )
}

export default Cart;
