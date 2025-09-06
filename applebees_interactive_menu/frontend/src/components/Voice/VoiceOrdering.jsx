import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useSpring, animated } from 'react-spring';
import { useSpeechRecognition } from 'react-speech-recognition';
import { useSpeechSynthesis } from 'react-speech-synthesis-hooks';

import './VoiceOrdering.css';

const VoiceOrdering = ({ 
  onOrderSubmit, 
  onVoiceCommand, 
  menuItems = [],
  isListening = false,
  onListeningChange 
}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [processing, setProcessing] = useState(false);
  const [orderItems, setOrderItems] = useState([]);
  const [currentOrder, setCurrentOrder] = useState({
    items: [],
    total: 0,
    specialInstructions: ''
  });

  // Speech recognition setup
  const {
    transcript: speechTranscript,
    listening,
    resetTranscript,
    browserSupportsSpeechRecognition
  } = useSpeechRecognition();

  // Speech synthesis setup
  const { speak, speaking, cancel } = useSpeechSynthesis();

  // Refs
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const microphoneRef = useRef(null);
  const animationFrameRef = useRef(null);

  // Animation states
  const [visualizationData, setVisualizationData] = useState([]);
  const [voiceLevel, setVoiceLevel] = useState(0);

  // Voice commands mapping
  const voiceCommands = {
    'order': ['order', 'get', 'want', 'like', 'add'],
    'quantities': ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'],
    'modifiers': ['spicy', 'mild', 'extra', 'no', 'with', 'without', 'light', 'heavy'],
    'sauces': ['buffalo', 'bbq', 'honey', 'sweet', 'asian', 'chile', 'ranch', 'blue cheese'],
    'actions': ['remove', 'cancel', 'clear', 'submit', 'finish', 'done', 'complete']
  };

  // Menu item aliases for voice recognition
  const menuAliases = {
    'wings': 'boneless_wings',
    'boneless wings': 'boneless_wings',
    'chicken wings': 'boneless_wings',
    'riblets': 'riblets',
    'ribs': 'riblets',
    'pork ribs': 'riblets',
    'salad': 'caesar_salad',
    'caesar': 'caesar_salad',
    'caesar salad': 'caesar_salad',
    'burger': 'classic_burger',
    'hamburger': 'classic_burger',
    'cheeseburger': 'classic_burger',
    'pasta': 'fettuccine_alfredo',
    'fettuccine': 'fettuccine_alfredo',
    'alfredo': 'fettuccine_alfredo'
  };

  useEffect(() => {
    if (speechTranscript !== transcript) {
      setTranscript(speechTranscript);
      processVoiceInput(speechTranscript);
    }
  }, [speechTranscript]);

  useEffect(() => {
    if (listening !== isRecording) {
      setIsRecording(listening);
      onListeningChange?.(listening);
      
      if (listening) {
        startAudioVisualization();
      } else {
        stopAudioVisualization();
      }
    }
  }, [listening]);

  const startAudioVisualization = useCallback(() => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    }

    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        microphoneRef.current = audioContextRef.current.createMediaStreamSource(stream);
        analyserRef.current = audioContextRef.current.createAnalyser();
        
        analyserRef.current.fftSize = 256;
        const bufferLength = analyserRef.current.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        microphoneRef.current.connect(analyserRef.current);
        
        const updateVisualization = () => {
          analyserRef.current.getByteFrequencyData(dataArray);
          
          // Calculate average volume
          const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
          const normalizedVolume = average / 255;
          
          setVoiceLevel(normalizedVolume);
          setVisualizationData(Array.from(dataArray));
          
          animationFrameRef.current = requestAnimationFrame(updateVisualization);
        };
        
        updateVisualization();
      })
      .catch(error => {
        console.error('Error accessing microphone:', error);
      });
  }, []);

  const stopAudioVisualization = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    
    if (microphoneRef.current) {
      microphoneRef.current.disconnect();
      microphoneRef.current = null;
    }
    
    setVoiceLevel(0);
    setVisualizationData([]);
  }, []);

  const processVoiceInput = useCallback((input) => {
    if (!input.trim()) return;

    setProcessing(true);
    
    // Normalize input
    const normalizedInput = input.toLowerCase().trim();
    
    // Parse voice input
    const parsedOrder = parseVoiceOrder(normalizedInput);
    
    if (parsedOrder) {
      updateOrder(parsedOrder);
      speakResponse(`Added ${parsedOrder.itemName} to your order`);
    } else {
      // Handle general commands
      handleGeneralCommands(normalizedInput);
    }
    
    setProcessing(false);
  }, [menuItems]);

  const parseVoiceOrder = useCallback((input) => {
    // Extract quantity
    let quantity = 1;
    const quantityMatch = input.match(/\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b/);
    if (quantityMatch) {
      const quantityText = quantityMatch[1];
      if (isNaN(quantityText)) {
        const numberWords = {
          'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
          'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        };
        quantity = numberWords[quantityText] || 1;
      } else {
        quantity = parseInt(quantityText);
      }
    }

    // Extract item name
    let itemName = '';
    let itemId = '';
    
    for (const [alias, id] of Object.entries(menuAliases)) {
      if (input.includes(alias)) {
        itemName = alias;
        itemId = id;
        break;
      }
    }

    if (!itemId) {
      // Try to find in menu items
      for (const item of menuItems) {
        if (input.includes(item.name.toLowerCase())) {
          itemName = item.name;
          itemId = item.id;
          break;
        }
      }
    }

    if (!itemId) return null;

    // Extract modifiers
    const modifiers = [];
    const sauces = [];
    
    // Check for sauces
    for (const sauce of voiceCommands.sauces) {
      if (input.includes(sauce)) {
        sauces.push(sauce);
      }
    }

    // Check for modifiers
    for (const modifier of voiceCommands.modifiers) {
      if (input.includes(modifier)) {
        modifiers.push(modifier);
      }
    }

    // Extract special instructions
    const specialInstructions = extractSpecialInstructions(input);

    return {
      itemId,
      itemName,
      quantity,
      modifiers,
      sauces,
      specialInstructions
    };
  }, [menuItems]);

  const extractSpecialInstructions = useCallback((input) => {
    const instructions = [];
    
    if (input.includes('well done') || input.includes('well-done')) {
      instructions.push('Well done');
    }
    
    if (input.includes('medium rare') || input.includes('medium-rare')) {
      instructions.push('Medium rare');
    }
    
    if (input.includes('no onions') || input.includes('without onions')) {
      instructions.push('No onions');
    }
    
    if (input.includes('extra cheese')) {
      instructions.push('Extra cheese');
    }
    
    if (input.includes('on the side')) {
      instructions.push('Sauce on the side');
    }
    
    return instructions.join(', ');
  }, []);

  const updateOrder = useCallback((parsedOrder) => {
    setCurrentOrder(prevOrder => {
      const existingItemIndex = prevOrder.items.findIndex(
        item => item.itemId === parsedOrder.itemId
      );

      let newItems;
      if (existingItemIndex >= 0) {
        // Update existing item
        newItems = [...prevOrder.items];
        newItems[existingItemIndex] = {
          ...newItems[existingItemIndex],
          quantity: newItems[existingItemIndex].quantity + parsedOrder.quantity,
          modifiers: [...new Set([...newItems[existingItemIndex].modifiers, ...parsedOrder.modifiers])],
          sauces: [...new Set([...newItems[existingItemIndex].sauces, ...parsedOrder.sauces])]
        };
      } else {
        // Add new item
        newItems = [...prevOrder.items, parsedOrder];
      }

      // Calculate total
      const total = newItems.reduce((sum, item) => {
        const menuItem = menuItems.find(mi => mi.id === item.itemId);
        return sum + (menuItem?.price || 0) * item.quantity;
      }, 0);

      return {
        ...prevOrder,
        items: newItems,
        total
      };
    });
  }, [menuItems]);

  const handleGeneralCommands = useCallback((input) => {
    if (input.includes('submit') || input.includes('finish') || input.includes('done')) {
      submitOrder();
    } else if (input.includes('clear') || input.includes('cancel')) {
      clearOrder();
    } else if (input.includes('remove')) {
      // Handle item removal
      const removeMatch = input.match(/remove\s+(.+)/);
      if (removeMatch) {
        const itemToRemove = removeMatch[1];
        removeItem(itemToRemove);
      }
    } else if (input.includes('total') || input.includes('how much')) {
      speakResponse(`Your total is $${currentOrder.total.toFixed(2)}`);
    } else if (input.includes('what') && input.includes('order')) {
      speakOrderSummary();
    } else {
      speakResponse("I didn't understand that. Please try again.");
    }
  }, [currentOrder]);

  const submitOrder = useCallback(() => {
    if (currentOrder.items.length === 0) {
      speakResponse("Your order is empty. Please add some items first.");
      return;
    }

    onOrderSubmit?.(currentOrder);
    speakResponse("Order submitted successfully! Your food will be ready soon.");
    clearOrder();
  }, [currentOrder, onOrderSubmit]);

  const clearOrder = useCallback(() => {
    setCurrentOrder({
      items: [],
      total: 0,
      specialInstructions: ''
    });
    speakResponse("Order cleared.");
  }, []);

  const removeItem = useCallback((itemName) => {
    setCurrentOrder(prevOrder => {
      const newItems = prevOrder.items.filter(item => 
        !item.itemName.toLowerCase().includes(itemName.toLowerCase())
      );
      
      const total = newItems.reduce((sum, item) => {
        const menuItem = menuItems.find(mi => mi.id === item.itemId);
        return sum + (menuItem?.price || 0) * item.quantity;
      }, 0);

      return {
        ...prevOrder,
        items: newItems,
        total
      };
    });
    
    speakResponse(`Removed ${itemName} from your order.`);
  }, [menuItems]);

  const speakResponse = useCallback((text) => {
    speak({
      text,
      voice: window.speechSynthesis.getVoices().find(voice => voice.name.includes('US')) || null,
      rate: 0.9,
      pitch: 1.0
    });
  }, [speak]);

  const speakOrderSummary = useCallback(() => {
    if (currentOrder.items.length === 0) {
      speakResponse("Your order is empty.");
      return;
    }

    const summary = currentOrder.items.map(item => 
      `${item.quantity} ${item.itemName}`
    ).join(', ');
    
    speakResponse(`Your order includes: ${summary}. Total: $${currentOrder.total.toFixed(2)}`);
  }, [currentOrder, speakResponse]);

  const startListening = useCallback(() => {
    if (!browserSupportsSpeechRecognition) {
      alert('Speech recognition is not supported in this browser.');
      return;
    }

    resetTranscript();
    // The useSpeechRecognition hook will handle starting/stopping
  }, [browserSupportsSpeechRecognition, resetTranscript]);

  const stopListening = useCallback(() => {
    // The useSpeechRecognition hook will handle stopping
  }, []);

  // Animation for voice level visualization
  const voiceLevelAnimation = useSpring({
    height: `${voiceLevel * 100}%`,
    opacity: isRecording ? 1 : 0,
    config: { tension: 300, friction: 20 }
  });

  const pulseAnimation = useSpring({
    scale: isRecording ? 1.2 : 1,
    config: { tension: 300, friction: 20 }
  });

  return (
    <div className="voice-ordering-container">
      {/* Voice Recognition Status */}
      <div className="voice-status">
        <animated.div 
          className={`voice-indicator ${isRecording ? 'recording' : ''}`}
          style={pulseAnimation}
        >
          <div className="voice-icon">
            {isRecording ? '🎤' : '🎤'}
          </div>
          
          <animated.div 
            className="voice-level-bar"
            style={voiceLevelAnimation}
          />
        </animated.div>

        <div className="status-text">
          {isRecording ? 'Listening...' : 'Tap to speak'}
        </div>
      </div>

      {/* Voice Controls */}
      <div className="voice-controls">
        <button
          className={`voice-btn ${isRecording ? 'recording' : ''}`}
          onClick={isRecording ? stopListening : startListening}
          disabled={processing}
        >
          {isRecording ? 'Stop Listening' : 'Start Voice Order'}
        </button>

        <button
          className="clear-btn"
          onClick={clearOrder}
          disabled={currentOrder.items.length === 0}
        >
          Clear Order
        </button>

        <button
          className="submit-btn"
          onClick={submitOrder}
          disabled={currentOrder.items.length === 0}
        >
          Submit Order
        </button>
      </div>

      {/* Transcript Display */}
      {transcript && (
        <div className="transcript-container">
          <h3>What you said:</h3>
          <div className="transcript-text">
            {transcript}
          </div>
          {confidence > 0 && (
            <div className="confidence">
              Confidence: {Math.round(confidence * 100)}%
            </div>
          )}
        </div>
      )}

      {/* Current Order */}
      <div className="current-order">
        <h3>Your Order</h3>
        {currentOrder.items.length === 0 ? (
          <p className="empty-order">No items in your order yet.</p>
        ) : (
          <div className="order-items">
            {currentOrder.items.map((item, index) => (
              <div key={index} className="order-item">
                <div className="item-info">
                  <span className="item-name">{item.itemName}</span>
                  <span className="item-quantity">x{item.quantity}</span>
                </div>
                
                {item.modifiers.length > 0 && (
                  <div className="item-modifiers">
                    Modifiers: {item.modifiers.join(', ')}
                  </div>
                )}
                
                {item.sauces.length > 0 && (
                  <div className="item-sauces">
                    Sauces: {item.sauces.join(', ')}
                  </div>
                )}
                
                {item.specialInstructions && (
                  <div className="item-instructions">
                    Special: {item.specialInstructions}
                  </div>
                )}
              </div>
            ))}
            
            <div className="order-total">
              <strong>Total: ${currentOrder.total.toFixed(2)}</strong>
            </div>
          </div>
        )}
      </div>

      {/* Voice Commands Help */}
      <div className="voice-help">
        <h3>Voice Commands</h3>
        <div className="commands-grid">
          <div className="command-category">
            <h4>Order Items</h4>
            <ul>
              <li>"I want boneless wings"</li>
              <li>"Add 2 riblets"</li>
              <li>"Get me a caesar salad"</li>
            </ul>
          </div>
          
          <div className="command-category">
            <h4>Modifiers</h4>
            <ul>
              <li>"Extra spicy"</li>
              <li>"No onions"</li>
              <li>"Sauce on the side"</li>
            </ul>
          </div>
          
          <div className="command-category">
            <h4>Actions</h4>
            <ul>
              <li>"Submit order"</li>
              <li>"Clear order"</li>
              <li>"What's my total?"</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Audio Visualization */}
      {isRecording && (
        <div className="audio-visualization">
          <div className="visualization-bars">
            {visualizationData.slice(0, 20).map((value, index) => (
              <div
                key={index}
                className="visualization-bar"
                style={{
                  height: `${(value / 255) * 100}%`,
                  backgroundColor: `hsl(${120 + (value / 255) * 60}, 70%, 50%)`
                }}
              />
            ))}
          </div>
        </div>
      )}

      {/* Processing Indicator */}
      {processing && (
        <div className="processing-indicator">
          <div className="spinner"></div>
          <p>Processing your voice command...</p>
        </div>
      )}
    </div>
  );
};

export default VoiceOrdering; 