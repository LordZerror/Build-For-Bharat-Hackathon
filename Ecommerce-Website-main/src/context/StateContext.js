import React, { createContext, useContext, useState} from 'react';
import { toast } from 'react-hot-toast';

const Context = createContext();

export const StateContext = ({ children }) => {
    const [cartItems , setCartItems] = useState([]);
    const [totalPrice , setTotalPrice] = useState(0);
    const [totalQuantities , setTotalQuantities] = useState(0);
    const [quantity, setquantity] = useState(1);
    const [asc, setasc] = useState(false);
    const [desc, setdesc] = useState(false);
    const [selectedBrands, setSelectedBrands] = useState([]);

  let foundProduct;
  let index;


  function onAdd(product, quantity){
    const checkProductInCart = cartItems.find((item) => item.id === product.id);
    
    setTotalPrice((prevTotalPrice) => prevTotalPrice + product.price * quantity);
    setTotalQuantities((prevTotalQuantities) => prevTotalQuantities + quantity);
    
    if(checkProductInCart) {
      const updatedCartItems = cartItems.map((cartProduct) => {
        if(cartProduct.id === product.id) return {
          ...cartProduct,
          quantity: cartProduct.quantity + quantity
        }
      })

      setCartItems(updatedCartItems);
    } else {
      product.quantity = quantity;
      
      setCartItems([...cartItems, { ...product }]);
    }

    toast.success(`${quantity} ${product.title} added to the cart.`);
    setquantity(1);
  } 

  const onRemove = (product) => {
    foundProduct = cartItems.find((item) => item.id === product.id);
    const newCartItems = cartItems.filter((item) => item.id !== product.id);

    setTotalPrice((prevTotalPrice) => prevTotalPrice -foundProduct.price * foundProduct.quantity);
    setTotalQuantities(prevTotalQuantities => prevTotalQuantities - foundProduct.quantity);
    setCartItems(newCartItems);
  }

  const toggleCartItemQuanitity = (id, value) => {
    foundProduct = cartItems.find((item) => item.id === id)
    index = cartItems.findIndex((product) => product.id === id);
    const newCartItems = cartItems.filter((item) => item.id !== id)

    if(value === 'inc') {
      setCartItems([...newCartItems, { ...foundProduct, quantity: foundProduct.quantity + 1 } ]);
      setTotalPrice((prevTotalPrice) => prevTotalPrice + foundProduct.price)
      setTotalQuantities(prevTotalQuantities => prevTotalQuantities + 1)
    } else if(value === 'dec') {
      if (foundProduct.quantity > 1) {
        setCartItems([...newCartItems, { ...foundProduct, quantity: foundProduct.quantity - 1 } ]);
        setTotalPrice((prevTotalPrice) => prevTotalPrice - foundProduct.price)
        setTotalQuantities(prevTotalQuantities => prevTotalQuantities - 1)
      }
    }
  }

  function increment(){
    setquantity(quantity + 1);
}

function decrement(){
    if(quantity>1)
    {
      setquantity(quantity - 1);
    }
}

function ascending(){
  setasc(true);
  setdesc(false);
}

function descending(){
  setdesc(true);
  setasc(false);
}

const handleClickOnSelectedProducts = (brand) => {
  if (selectedBrands.includes(brand)) {
    setSelectedBrands(selectedBrands.filter(item => item !== brand)); 
  } else {
    setSelectedBrands([...selectedBrands, brand]); 
  }
};


  return (
    <Context.Provider
      value={{
        asc,
        desc,
        cartItems,
        totalPrice,
        totalQuantities,
        quantity,
        selectedBrands,
        increment,
        decrement,
        onAdd,
        toggleCartItemQuanitity,
        onRemove,
        setCartItems,
        setTotalPrice,
        setTotalQuantities,
        setasc,
        setdesc,
        ascending,
        descending,
        setSelectedBrands,
        handleClickOnSelectedProducts
      }}
    >
      {children}
    </Context.Provider>
  )
}

export const useStateContext = () => useContext(Context);