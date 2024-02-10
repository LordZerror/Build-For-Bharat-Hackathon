import React from 'react'
import Navbar from './Navbar'
import Grid from'./Products/Grid';
import Sidebar from './Sidebar'
import '../CSS/homepage.css'

function HomePage() {
  return (
    <>
     <div>
      <Navbar />
    </div>
    <div className='main--content'>
      <Sidebar />
      <Grid />
    </div>
    </>
  );
}

export default HomePage;