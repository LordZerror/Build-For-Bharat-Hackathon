import React, { useState, useEffect } from 'react';
import '../CSS/navbar.css';
import searchButton from '../img/search-button.png';
import authicon from '../img/auth.png';
import cart from '../img/cart.png';
import { Link } from 'react-router-dom';
import { useStateContext } from '../context/StateContext';
import { onAuthStateChanged  } from 'firebase/auth';
import { auth } from '../firebase';

function Header() {
  const { totalQuantities } = useStateContext();
  const [user, setUser] = useState(null);

  useEffect(() => {
    onAuthStateChanged(auth, (user) => {
        if (user) {
            console.log(user.uid)
           setUser(user)
        } else {
            setUser(null)
        }
    });
}, []); 


  return (
    <div className='navbar'>
      <Link to='/' style={{ textDecoration: 'none' }}>
        <div>
          <h4 className='logo'>WIRE CART</h4>
        </div>
      </Link>
      <div className='search--section'>
        <input className='search-bar' placeholder='Search for products' type='text'></input>
        <button className='search-button'>
          <img src={searchButton} className='search-button-icon'></img>
        </button>
      </div>
      <div className='header--buttons'>
        <div className='auth--button'>
          <button className='auth'>
            <img src={authicon}></img>
            <h6 className='auth-annotation'>{user ? user.email : 'Profile'}</h6>
          </button>
          {user ? (
            <div className='login-register-option'>
              <ul className='unordered-list'>
                <li className='list' onClick={() => auth.signOut()}>
                  Sign Out
                </li>
                <Link to='/OrderHistory'>
                  <li className='list'>Order History</li>
                </Link>
              </ul>
            </div>
          ) : (
            <div className='login-register-option'>
              <ul className='unordered-list'>
                <Link to='/Login'>
                  <li className='list'>Login</li>
                </Link>
                <Link to='/Register'>
                  <li className='list'>Register</li>
                </Link>
              </ul>
            </div>
          )}
        </div>
        <div className='cart--button'>
          <Link to='/My-Cart' style={{ textDecoration: 'none' }}>
            <button className='cart'>
              <img src={cart}></img>
              <h6 className='cart-annotation'>Cart</h6>
            </button>
            <div className='counter'>{totalQuantities}</div>
          </Link>
        </div>
      </div>
    </div>
  );
}

export default Header;
