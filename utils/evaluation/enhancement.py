import seaborn as sns
import pandas as pd
import numpy as np

from ..plotting.maps import counts as mapcount
from ..plotting.maps import flow

def check_uids(bbh, max_timespan_hours=4):
    gr = bbh.reset_index().groupby('uid')

    # How many bbh entries per uid?
    # Should be, like, less than 10 or something.
    c = bbh.uid.value_counts()
    if any(c > 10):
        print('{:,} uids have more than 10 entries'.format(
            (c>10).sum()))
        print(c.head(5))
        print()
    else:
        print('uid entry counts look OK')
        print()

    # How much time does each uid span?
    # Should be less than max_timespan
    ts = (gr.Timestamp.max() - gr.Timestamp.min())/pd.to_timedelta('1 hour')
    print('Median timespan for each uid: {:.2f} hours'.format(ts.median()))
    print('Min timespan for a uid: {:.2f} hours'.format(ts.min()))
    print('Max timespan for a uid: {:.2f} hours (index {})'.format(
        ts.max(), ts.argmax()))
    c = (ts>max_timespan_hours).sum()
    print('{} uids exceed {} hours'.format(
        c, max_timespan_hours))
    if c == 0:
        print('Looks OK')
    print()

    # How many runs per hangerid?
    uids_per_hangerid = bbh.uid.groupby(bbh.HangerID).nunique()
    print('On average, {:.5f} uids per'
          ' hangerid'.format(uids_per_hangerid.mean()))
    print('Max uids for a hangerid: {:.2f} (hangerid {})'.format(
        uids_per_hangerid.max(), uids_per_hangerid.argmax()))
    print('(std: {:.3f})'.format(uids_per_hangerid.std()))
    print()

    # How many hangerids per uid?
    # Should be 1
    c = gr.HangerID.nunique()
    if c.mean() != 1 or c.std() != 0:
        print('On average, {:.5f} hangerids per'
              ' uid (should be 1)'.format(c.mean()))
        print('(std: {})'.format(c.std()))
    else:
        print('hangerids per uid looks OK')
    print()

def start_and_end_points_plot(bbh, logtrans=False, subset=None):
    title='Start and end points for bbh'

    gr = bbh.reset_index().groupby('uid')

    starts = gr.head(1).Tx.value_counts()
    ends = gr.tail(1).Tx.value_counts()
    if logtrans:
        starts = np.log10(1+starts).astype(int)
        ends = np.log10(1+ends).astype(int)
        title += ' (log10 transformed)'
    ax = mapcount.plot(starts.to_dict(), position='left',
                       title=title,
                       subset=subset)

    mapcount.plot(ends.to_dict(), ax=ax, position='right',
                  barcolor='#FFAAAA')

    return ax

def inspect_reinspected(bbh, reinspected, **flow_kwargs):
    reinspected = reinspected.reset_index()
    yes = reinspected.uid[reinspected.reinspected == 1].values
    no = reinspected.uid[reinspected.reinspected == 0].values

    ax = flow.plot(bbh[np.in1d(bbh.uid, yes)], color='k', **flow_kwargs)
    ax = flow.plot(bbh[np.in1d(bbh.uid, no)], color='g', ax=ax, **flow_kwargs)

    ax.set_title('{:,} hanger tracks ({:,} reinspected)'.format(
        reinspected.uid.nunique(), len(yes)))

    return ax

