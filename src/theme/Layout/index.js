import React from 'react';
import {RagChatbot} from '../../components/RagChatbot/RagChatbot';
import {useLocation} from '@docusaurus/router';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import OriginalLayout from '@theme-original/Layout';
import {useEffect, useState} from 'react';

// This is a wrapper around the original Docusaurus Layout that adds the RAG chatbot
export default function Layout(props) {
  const location = useLocation();
  const {siteConfig} = useDocusaurusContext();
  const [showChatbot, setShowChatbot] = useState(true);

  // Determine if chatbot should be visible on this page
  useEffect(() => {
    // Hide chatbot on certain pages if needed (customize as needed)
    const hideOnPaths = ['/pathname-to-hide']; // Add specific paths where chatbot shouldn't appear
    const shouldHide = hideOnPaths.some(path => location.pathname.includes(path));
    setShowChatbot(!shouldHide);
  }, [location.pathname]);

  return (
    <>
      <OriginalLayout {...props}>
        {props.children}
        {showChatbot && (
          <div className="rag-chatbot-wrapper">
            <RagChatbot backendUrl={`${siteConfig.customFields?.backendUrl || '/api/v1'}`} />
          </div>
        )}
      </OriginalLayout>

      {/* Add custom styles for chatbot positioning */}
      <style jsx>{`
        .rag-chatbot-wrapper {
          position: fixed;
          bottom: 20px;
          right: 20px;
          width: 380px;
          height: 500px;
          z-index: 1000;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
          border-radius: 12px;
          overflow: hidden;
        }

        @media (max-width: 768px) {
          .rag-chatbot-wrapper {
            width: calc(100vw - 20px);
            height: 50vh;
            bottom: 10px;
            right: 10px;
          }
        }
      `}</style>
    </>
  );
}