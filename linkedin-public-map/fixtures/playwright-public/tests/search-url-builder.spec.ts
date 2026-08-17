// Offline tests for buildPublicPeopleSearchUrl() — verifies URN maps resolve and
// that JSON-array parameters are URL-encoded exactly as LinkedIn expects on the
// authenticated /search/results/people/ surface.

import { expect, test } from '@playwright/test';
import { buildPublicPeopleSearchUrl } from '../src/pages/SearchResultsPeoplePage.js';

function parse(url: string): URLSearchParams {
  return new URL(url).searchParams;
}

test.describe('buildPublicPeopleSearchUrl', () => {
  test('minimal call sets keywords and defaults origin=FACETED_SEARCH', () => {
    const url = buildPublicPeopleSearchUrl({ keywords: 'machine learning engineer' });
    expect(url.startsWith('https://www.linkedin.com/search/results/people/?')).toBe(true);
    const sp = parse(url);
    expect(sp.get('keywords')).toBe('machine learning engineer');
    expect(sp.get('origin')).toBe('FACETED_SEARCH');
  });

  test('geos resolves named country to GEO_URNS and JSON-encodes the array', () => {
    const url = buildPublicPeopleSearchUrl({
      keywords: 'staff engineer',
      geos: ['US', 'Canada'],
    });
    const sp = parse(url);
    expect(JSON.parse(sp.get('geoUrn')!)).toEqual(['103644278', '101174742']);
  });

  test('numeric geos pass through unchanged', () => {
    const url = buildPublicPeopleSearchUrl({
      keywords: 'pm',
      geos: [101282230, 105015875],
    });
    expect(JSON.parse(parse(url).get('geoUrn')!)).toEqual(['101282230', '105015875']);
  });

  test('industries resolves named industry to INDUSTRY_URNS', () => {
    const url = buildPublicPeopleSearchUrl({
      keywords: 'researcher',
      industries: ['ComputerSoftware', 'HigherEducation'],
    });
    expect(JSON.parse(parse(url).get('industry')!)).toEqual(['4', '68']);
  });

  test('currentCompany and pastCompany serialise as JSON string arrays', () => {
    const url = buildPublicPeopleSearchUrl({
      keywords: 'rust',
      currentCompanyUrns: [1035],
      pastCompanyUrns: [1441, 1586],
    });
    const sp = parse(url);
    expect(JSON.parse(sp.get('currentCompany')!)).toEqual(['1035']);
    expect(JSON.parse(sp.get('pastCompany')!)).toEqual(['1441', '1586']);
  });

  test('schoolFilter encodes the schoolUrns array', () => {
    const url = buildPublicPeopleSearchUrl({
      keywords: 'phd',
      schoolUrns: [10999, 18043],
    });
    expect(JSON.parse(parse(url).get('schoolFilter')!)).toEqual(['10999', '18043']);
  });

  test('network is a comma-joined letter list using NETWORK_CODES mapping', () => {
    const url = buildPublicPeopleSearchUrl({
      keywords: 'cto',
      network: ['second_degree', 'third_plus_degree'],
    });
    expect(parse(url).get('network')).toBe('S,O');
  });

  test('first-degree only translates to F', () => {
    const url = buildPublicPeopleSearchUrl({
      keywords: 'recruiter',
      network: ['first_degree'],
    });
    expect(parse(url).get('network')).toBe('F');
  });

  test('boolean keywords with uppercase AND/OR/NOT are preserved literally (URL-encoded)', () => {
    const url = buildPublicPeopleSearchUrl({
      keywords: '("staff engineer" OR "principal engineer") AND (rust OR golang) NOT recruiter',
    });
    const sp = parse(url);
    // URLSearchParams decodes the value back for us
    expect(sp.get('keywords')).toBe(
      '("staff engineer" OR "principal engineer") AND (rust OR golang) NOT recruiter',
    );
  });

  test('first/last/title/profileLanguage pass through', () => {
    const url = buildPublicPeopleSearchUrl({
      keywords: 'ml',
      firstName: 'Ada',
      lastName: 'Lovelace',
      title: 'Research Scientist',
      profileLanguage: 'en',
    });
    const sp = parse(url);
    expect(sp.get('firstName')).toBe('Ada');
    expect(sp.get('lastName')).toBe('Lovelace');
    expect(sp.get('title')).toBe('Research Scientist');
    expect(sp.get('profileLanguage')).toBe('en');
  });

  test('omits optional params entirely when not provided (no empty geoUrn etc.)', () => {
    const url = buildPublicPeopleSearchUrl({ keywords: 'devops' });
    const sp = parse(url);
    expect(sp.has('geoUrn')).toBe(false);
    expect(sp.has('industry')).toBe(false);
    expect(sp.has('network')).toBe(false);
    expect(sp.has('currentCompany')).toBe(false);
    expect(sp.has('pastCompany')).toBe(false);
    expect(sp.has('schoolFilter')).toBe(false);
  });

  test('caller may override origin', () => {
    const url = buildPublicPeopleSearchUrl({
      keywords: 'dev',
      origin: 'GLOBAL_SEARCH_HEADER',
    });
    expect(parse(url).get('origin')).toBe('GLOBAL_SEARCH_HEADER');
  });
});
