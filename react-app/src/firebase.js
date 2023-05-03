// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getStorage } from "firebase/storage";
import { getFirestore } from "firebase/firestore";

// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyARVQ6ErRCEE5Iq-cvv9HD3nKaB95_jn48",
    authDomain: "bigco-studio.firebaseapp.com",
    projectId: "bigco-studio",
    storageBucket: "bigco-studio.appspot.com",
    messagingSenderId: "723040240101",
    appId: "1:723040240101:web:ad93a335c61e9436336f0c"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const storage = getStorage(app)

export const db = getFirestore(app)