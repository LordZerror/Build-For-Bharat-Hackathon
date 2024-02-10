import React, { useState } from 'react';
import '../CSS/ProductFilteringButtons.css';
import data from '../components/Products/data';
import { useStateContext } from '../context/StateContext'

const ProductFilteringButtons = () => {
  const {selectedBrands,handleClickOnSelectedProducts} = useStateContext();
  
  const uniqueBrands = [...new Set(data.map(product => product.brand))];
  uniqueBrands.sort();

  return (
    <div className='rowwise-container'>
      {uniqueBrands.map((brand, index) => (
        <button 
          key={index} 
          className={selectedBrands.includes(brand) ? 'brand-button selected' : 'brand-button'}
          onClick={() => handleClickOnSelectedProducts(brand)}
        >
          {brand}
        </button>
      ))}
    </div>
  );
};

export default ProductFilteringButtons;
