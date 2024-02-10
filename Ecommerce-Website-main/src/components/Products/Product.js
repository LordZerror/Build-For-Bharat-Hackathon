import React from 'react'
import {Link} from 'react-router-dom'
import '../../CSS/product.css'
import Rating from '@mui/material/Rating';

function Products({ products, loading }) {
    if (loading) {
      return <h2>Loading...</h2>;
    }
   
    return (
      <div className='product--grid'>
        {products.map(product =>(
          <Link to={`/${product.id}`} style={{ textDecoration: 'none' }}>
          <div className='product--block' key={product.id}>
            <div>
              <img className='product--thumbnail' src={product.thumbnail}></img> 
            </div>
            <div className='product--details'>
              <h3 className='product--title'>{product.title}</h3>
              <Rating className='product--ratings' name="read-only" value={(Math.floor(Math.random() * (25 - 15 + 1)) + 15)/5} readOnly />
              <p className='product--price'>${product.price}</p>
            </div>
          </div>
          </Link>
        ))}
       </div>
    );
    

}
export default Products;