import React, { useState } from 'react';
import { useLocation } from 'react-router-dom';
import products from './Products/data';
import Navbar from './Navbar';
import '../CSS/productslayout.css';
import { useStateContext } from '../context/StateContext';
import { Toaster } from 'react-hot-toast';

function ProductsLayout() {
  const location = useLocation();
  const productId = Number(location.pathname.substring(1));
  const singleProduct = products.find(product => product.id === productId);

  const { decrement, increment, quantity, onAdd } = useStateContext();
  const [mainImage, setMainImage] = useState(singleProduct.thumbnail);

  const handleThumbnailClick = (image) => {
    setMainImage(image);
  };

  return (
    <div>
      <div><Toaster /></div>
      <Navbar />
      <div className='product---details'>
        <div className='product-display'>
          <img src={mainImage} className='product-image' alt='Product' />
          <div className='thumbnail-container'>
            {singleProduct.images.map((image, index) => (
              <img
                key={index}
                src={image}
                className='thumbnail-image'
                alt={`Thumbnail ${index + 1}`}
                onClick={() => handleThumbnailClick(image)}
              />
            ))}
          </div>
        </div>
        <div className='product-info'>
          <div>
            <h1 className='productlayout--title'>{singleProduct.title}</h1>
          </div>
          <div className='about-product'>
            <b>Product description :</b>
          </div>
          <div>
            <i className='product-desc'>{singleProduct.description}</i>
          </div>
          <div className='quantity'>
            <p className='qty-label'>Qty :</p>
            <button className='decrement-qty' onClick={decrement}>-</button>
            <span className='qty-counter'>{quantity}</span>
            <button className='increment-qty' onClick={increment}>+</button>
          </div>
          <div className='product-totalcost'>
            <p className='product-price'>${singleProduct.price}</p>
          </div>
          <div className='button-for-cart'>
            <button className='add-to-cart' onClick={() => onAdd(singleProduct, quantity)}>Add to Cart</button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ProductsLayout;
