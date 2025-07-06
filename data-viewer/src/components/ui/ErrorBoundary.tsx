import React from 'react';

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
  errorInfo?: React.ErrorInfo;
}

interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
  resetKey?: string | number;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    // Update state so the next render will show the fallback UI
    return { hasError: true, error };
  }

  componentDidUpdate(prevProps: ErrorBoundaryProps) {
    // Reset error state when resetKey changes
    if (prevProps.resetKey !== this.props.resetKey && this.state.hasError) {
      this.setState({ hasError: false });
    }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // Store error info for potential debugging
    this.setState({ errorInfo });
    
    // Call optional error handler
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
    
    // Only log non-render errors to avoid spam
    if (!error.message.includes('Too many re-renders')) {
      console.warn('Error caught by boundary:', {
        message: error.message,
        stack: error.stack,
        componentStack: errorInfo.componentStack
      });
      
      // In production, you might want to send this to a monitoring service
      if (import.meta.env.PROD) {
        // Example: sendErrorToMonitoring(error, errorInfo);
      }
    }
  }

  render() {
    if (this.state.hasError) {
      // Return custom fallback UI if provided
      if (this.props.fallback) {
        return this.props.fallback;
      }
      
      // Default fallback UI for development
      if (import.meta.env.DEV && this.state.error) {
        return (
          <div style={{
            padding: '20px',
            margin: '20px',
            border: '2px solid #f87171',
            borderRadius: '8px',
            backgroundColor: 'rgba(254, 226, 226, 0.1)',
            color: '#991b1b'
          }}>
            <h3>Component Error</h3>
            <p><strong>Error:</strong> {this.state.error.message}</p>
            <details style={{ marginTop: '10px' }}>
              <summary>Error Details</summary>
              <pre style={{ 
                fontSize: '12px', 
                overflow: 'auto', 
                marginTop: '8px',
                padding: '8px',
                backgroundColor: 'rgba(0,0,0,0.1)',
                borderRadius: '4px'
              }}>
                {this.state.error.stack}
              </pre>
            </details>
            <button 
              onClick={() => this.setState({ hasError: false, error: undefined, errorInfo: undefined })}
              style={{
                marginTop: '10px',
                padding: '6px 12px',
                backgroundColor: '#dc2626',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer'
              }}
            >
              Try Again
            </button>
          </div>
        );
      }
      
      // Return nothing in production for image loading errors
      return null;
    }

    return this.props.children;
  }
}