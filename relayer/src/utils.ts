import { EventRecord } from "@polkadot/types/interfaces/system";
import { filter } from "rxjs/operators";
import { Observable, pipe, UnaryFunction, OperatorFunction } from 'rxjs';
import { ParaHeadAndId, ParachainsMap } from "./types";
import Parachain from "./parachain";
import Target from "./target";

// TODO: consider moving to a separate utils module
// TODO: implement tests
export const getParaHeadAndIdFromRecord = ({ event }: EventRecord): ParaHeadAndId => {
    // use 'any' because this is not typed array - element can be number, string or Record<string, unknown>
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { paraHead, paraId } = (event.toJSON().data as Array<any>)[0]
        .descriptor;

    return { paraHead, paraId };
};

// TODO: more explicit function name
export const isRelevantRecord =
    (index: number) =>
        ({ phase, event }: EventRecord): boolean => {
            return (
                // filter the specific events based on the phase and then the
                // index of our extrinsic in the block
                phase.isApplyExtrinsic &&
                phase.asApplyExtrinsic.eq(index) &&
                event.section == "paraInclusion" &&
                event.method == "CandidateIncluded"
            );
        };

export const createParachainsMap = async (target: Target, parachains: {
    url: string,
    paraId: number
}[]): Promise<ParachainsMap> => {
    const map = new Map();

    for await (const parachain of parachains) {
        const feedId = await target.sendCreateFeedTx();
        const chain = new Parachain({ feedId, url: parachain.url });
        map.set(parachain.paraId, chain)
    }

    return map;
}

export function filterNullish<T>(): UnaryFunction<Observable<T | null | undefined>, Observable<T>> {
    return pipe(
        filter(x => x != null) as OperatorFunction<T | null | undefined, T>
    );
}
