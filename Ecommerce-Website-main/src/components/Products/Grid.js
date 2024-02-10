import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Products from './Product';
import Pagination from '../Pagination';
import { useStateContext } from '../../context/StateContext';

function Grid() {
  const { asc, desc, selectedBrands } = useStateContext();

  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [productsPerPage] = useState(10);

  useEffect(() => {
    const fetchProducts = async () => {
      setLoading(true);
      const res = await axios.get('https://dummyjson.com/products');
      setProducts(res.data.products);
      setLoading(false);
    };

    fetchProducts();
  }, []);

  useEffect(() => {
    if (asc === true) {
      setProducts(prevProducts => [...prevProducts].sort((a, b) => parseFloat(a.price) - parseFloat(b.price)));
    }

    if (desc === true) {
      setProducts(prevProducts => [...prevProducts].sort((a, b) => parseFloat(b.price) - parseFloat(a.price)));
    }
  }, [asc, desc]);

  const filteredProducts = selectedBrands.length > 0 ? products.filter(product => selectedBrands.includes(product.brand)) : products;

  const indexOfLastProduct = currentPage * productsPerPage;
  const indexOfFirstProduct = indexOfLastProduct - productsPerPage;
  const currentProducts = filteredProducts.slice(indexOfFirstProduct, indexOfLastProduct);

  const paginate = pageNumber => setCurrentPage(pageNumber);

  return (
    <div>
      <Products products={currentProducts} loading={loading} />
      <Pagination productsPerPage={productsPerPage} totalProducts={filteredProducts.length} paginate={paginate} />
    </div>
  );
}

export default Grid;
