import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className="hero-content">
          <div className="hero-text">
            <Heading as="h1" className="hero__title">
              {siteConfig.title}
            </Heading>
            <p className="hero__subtitle">{siteConfig.tagline}</p>
            <div className={styles.buttons}>
              <Link
                className="button button--secondary button--lg"
                to="/docs/intro">
                Start Learning 
              </Link>
              <Link
                className="button button--primary button--lg margin-left--md"
                to="/docs/module-1-ros2">
                Jump to Module 1
              </Link>
            </div>
          </div>
          <div className="hero-visual">
            <div className={styles.robotAnimation}>
              <svg viewBox="0 0 200 200" className={styles.robotSvg}>
                <circle cx="100" cy="70" r="20" fill="#444" />
                <rect x="85" y="90" width="30" height="40" fill="#666" />
                <rect x="75" y="130" width="10" height="30" fill="#888" />
                <rect x="115" y="130" width="10" height="30" fill="#888" />
                <circle cx="90" cy="65" r="3" fill="#ff6b6b" />
                <circle cx="110" cy="65" r="3" fill="#ff6b6b" />
                <rect x="95" y="75" width="10" height="2" fill="#444" />
              </svg>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

function QuickNavigation() {
  return (
    <section className={styles.quickNav}>
      <div className="container">
        <div className="text--center padding-bottom--lg">
          <h2>Start Learning Today</h2>
          <p className={styles.navDescription}>
            Jump directly to the module that interests you most, or follow the sequential path for comprehensive learning.
          </p>
        </div>

        <div className="row">
          <div className="col col--2 col--offset-1">
            <Link to="/docs/intro" className={styles.navCard}>
              <h3>Start Here</h3>
              <p>Begin your robotics journey</p>
            </Link>
          </div>
          <div className="col col--2">
            <Link to="/docs/module-1-ros2" className={styles.navCard}>
              <h3>Module 1</h3>
              <p>ROS 2 Fundamentals</p>
            </Link>
          </div>
          <div className="col col--2">
            <Link to="/docs/module-2-digital-twin" className={styles.navCard}>
              <h3>Module 2</h3>
              <p>Digital Twin & Simulation</p>
            </Link>
          </div>
          <div className="col col--2">
            <Link to="/docs/module-3-nvidia-isaac" className={styles.navCard}>
              <h3>Module 3</h3>
              <p>NVIDIA Isaac & AI</p>
            </Link>
          </div>
          <div className="col col--2">
            <Link to="/docs/module-4-vla" className={styles.navCard}>
              <h3>Module 4</h3>
              <p>VLA Models & Capstone</p>
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}

function ValueProposition() {
  return (
    <section className={styles.valueProposition}>
      <div className="container">
        <div className="text--center padding-bottom--lg">
          <h2>Why Learn Physical AI & Humanoid Robotics?</h2>
          <p className={styles.valueDescription}>
            The future of robotics lies at the intersection of artificial intelligence and physical systems.
            This comprehensive book bridges the gap between digital AI and embodied intelligence,
            preparing you for the next generation of intelligent robots.
          </p>
        </div>

        <div className="row">
          <div className="col col--4">
            <div className={styles.valueCard}>
              <h3>Cutting-Edge Skills</h3>
              <p>Master the latest technologies in ROS 2, NVIDIA Isaac, simulation, and Vision-Language-Action models that leading robotics companies use.</p>
            </div>
          </div>
          <div className="col col--4">
            <div className={styles.valueCard}>
              <h3>Practical Applications</h3>
              <p>Build real-world projects from simple ROS 2 nodes to complex humanoid robot control systems with AI integration.</p>
            </div>
          </div>
          <div className="col col--4">
            <div className={styles.valueCard}>
              <h3>Career Advancement</h3>
              <p>Position yourself at the forefront of robotics development, one of the fastest-growing fields in technology.</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function ModulePreview() {
  return (
    <section className={styles.modulePreview}>
      <div className="container">
        <div className="row">
          <div className="col col--4">
            <div className={styles.moduleCard}>
              <h3>Module 1: ROS 2</h3>
              <p>Learn the fundamentals of Robot Operating System 2, the nervous system of robotics.</p>
              <Link to="/docs/module-1-ros2" className={styles.moduleLink}>
                Explore Module 1
              </Link>
            </div>
          </div>
          <div className="col col--4">
            <div className={styles.moduleCard}>
              <h3>Module 2: Digital Twin</h3>
              <p>Master simulation with Gazebo and Unity for creating virtual replicas of physical robots.</p>
              <Link to="/docs/module-2-digital-twin" className={styles.moduleLink}>
                Explore Module 2
              </Link>
            </div>
          </div>
          <div className="col col--4">
            <div className={styles.moduleCard}>
              <h3>Module 3: NVIDIA Isaac</h3>
              <p>Implement AI-powered robotics with NVIDIA Isaac Sim and Isaac ROS for advanced applications.</p>
              <Link to="/docs/module-3-nvidia-isaac" className={styles.moduleLink}>
                Explore Module 3
              </Link>
            </div>
          </div>
        </div>
        <div className="row" style={{marginTop: '2rem'}}>
          <div className="col col--4 col--offset-2">
            <div className={styles.moduleCard}>
              <h3>Module 4: VLA Models</h3>
              <p>Integrate Vision-Language-Action models for cognitive robotics and human-robot interaction.</p>
              <Link to="/docs/module-4-vla" className={styles.moduleLink}>
                Explore Module 4
              </Link>
            </div>
          </div>
          <div className="col col--4">
            <div className={styles.moduleCard}>
              <h3>Capstone Project</h3>
              <p>Synthesize all modules in a comprehensive project integrating physical AI systems.</p>
              <Link to="/docs/module-4-vla/capstone" className={styles.moduleLink}>
                Start Capstone
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Physical AI & Humanoid Robotics`}
      description="A comprehensive guide to bridging digital AI with physical robotics">
      <HomepageHeader />
      <main>
        <QuickNavigation />
        <ValueProposition />
        <ModulePreview />
      </main>
    </Layout>
  );
}
