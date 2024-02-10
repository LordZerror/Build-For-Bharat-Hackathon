import {React, useState} from 'react'
import Navbar from '../Navbar'
import '../../CSS/login.css'
import { signInWithEmailAndPassword } from 'firebase/auth';
import { auth } from '../../firebase';
import { Toaster, toast } from 'react-hot-toast';

const Login = () => {

  const [email, setEmail] = useState('');
  const [pwd, setPwd] = useState('');

  function handleSubmit(e) {
    e.preventDefault();

    signInWithEmailAndPassword(auth, email, pwd)
      .then((userCredential) => {
        console.log(userCredential.user);
        toast.success(`Login successful`)
      })
      .catch((error) => {
        console.log(error.code)
        toast.error(`Login unsuccessful`)
      });
  }

  return (
    <>
        <div><Navbar/></div>
        <div><Toaster /></div>
        <div className='login--segment'>
            <form className='login--component' onSubmit={handleSubmit}>
               <div><h3 className='logo--login'>WIRE CART</h3></div>
               <div className='login--email-address'>
                <input placeholder='Email-address' className='e-mail' value={email} onChange={(e) => setEmail(e.target.value)}></input>
               </div>
               <div className='login--password'>
                <input placeholder='password' type='password' className='pwd' value={pwd} onChange={(e) => setPwd(e.target.value)}></input>
               </div>
               <div className='login--sign-up'>
                <button type='submit' className='login--button'>Log in</button>
               </div>
            </form>
        </div>
    </>
  )
}

export default Login