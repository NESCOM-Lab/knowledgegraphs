// import { useState } from 'react'
import './App.css'
import Chat from './Chat'

function App() {
  // const [count, setCount] = useState(0)

  return (
    <>
      <h2>Knowledge Graphs</h2>
      <form>
        <input type="text"></input>
        <input type="submit" value=""></input>

      </form>
      <Chat text={"LLM response..."}/>
      

    </>
  )
}

export default App
