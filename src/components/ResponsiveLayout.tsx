import React from 'react';
import Layout from '@theme/Layout';

interface ResponsiveLayoutProps {
  children: React.ReactNode;
  title?: string;
  description?: string;
  wrapperClassName?: string;
}

const ResponsiveLayout: React.FC<ResponsiveLayoutProps> = ({
  children,
  title,
  description,
  wrapperClassName
}) => {
  return (
    <Layout title={title} description={description}>
      <div className={`container ${wrapperClassName || ''}`} style={{
        padding: '2rem 1rem',
        maxWidth: '100%',
        width: '100%'
      }}>
        <div className="row" style={{
          display: 'flex',
          flexWrap: 'wrap',
          margin: '0 -0.5rem'
        }}>
          <div className="col" style={{
            flex: '1 1 0',
            minWidth: '300px',
            padding: '0 0.5rem'
          }}>
            {children}
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default ResponsiveLayout;