import {React,useState} from 'react'
import '../CSS/sidebar.css'
import ProductFilteringButtons from './ProductFilteringButtons'
import { useStateContext } from '../context/StateContext'

export const Sidebar = () => {
  const {ascending,descending} = useStateContext();

  return (
    <div className='sidebar'>
      <ProductFilteringButtons />
      <div className='sorting--filtering'>
        <div className='sorting'>
          <button className='low--high' onClick={ascending}>Low to High</button>
          <button className='high--low' onClick={descending}>High to Low</button>
        </div>
      </div>
    </div>
  )
}

export default Sidebar;
